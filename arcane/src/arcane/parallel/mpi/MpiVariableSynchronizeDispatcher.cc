// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiVariableSynchronizeDispatcher.cc                         (C) 2000-2025 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiTimeInterval.h"
#include "arcane/parallel/IStat.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/impl/IDataSynchronizeImplementation.h"

#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Le fonctionnement de l'algorithme de synchronisation est le suivant. Les
 * trois premiers points sont dans beginSynchronize() et les deux derniers dans
 * endSynchronize(). Le code actuel ne permet qu'un synchronisation non
 * bloquante à la fois.
 *
 * 1. Poste les messages de réception
 * 2. Recopie dans les buffers d'envoi les valeurs à envoyer. On le fait après
 *    avoir posté les messages de réception pour faire un peu de recouvrement
 *    entre le calcul et les communications.
 * 3. Poste les messages d'envoi.
 * 4. Fait un WaitSome sur les messages de réception. Dès qu'un message arrive,
 *    on recopie le buffer de réception dans le tableau de la variable. On
 *    peut simplifier le code en faisant un WaitAll et en recopiant à la fin
 *    toutes les valeurs.
 * 5. Fait un WaitAll des messages d'envoi pour libérer les requêtes.
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation optimisée pour MPI de la synchronisation.
 *
 * Par rapport à la version de base, cette implémentation fait un MPI_Waitsome
 * (au lieu d'un Waitall) et recopie dans le buffer de destination
 * dès qu'un message arrive.
 *
 * NOTE: cette optimisation respecte la norme MPI qui dit qu'on ne doit
 * plus toucher à la zone mémoire d'un message tant que celui-ci n'est
 * pas fini.
 */
class MpiVariableSynchronizeDispatcher
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit MpiVariableSynchronizeDispatcher(Factory* f);

 protected:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* ds_buf) override;
  void endSynchronize(IDataSynchronizeBuffer* ds_buf) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng;
  UniqueArray<Parallel::Request> m_original_recv_requests;
  UniqueArray<bool> m_original_recv_requests_done;
  Ref<Parallel::IRequestList> m_receive_request_list;
  Ref<Parallel::IRequestList> m_send_request_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiVariableSynchronizeDispatcher::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  explicit Factory(MpiParallelMng* mpi_pm)
  : m_mpi_parallel_mng(mpi_pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new MpiVariableSynchronizeDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiVariableSynchronizerFactory(MpiParallelMng* mpi_pm)
{
  auto* x = new MpiVariableSynchronizeDispatcher::Factory(mpi_pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiVariableSynchronizeDispatcher::
MpiVariableSynchronizeDispatcher(Factory* f)
: m_mpi_parallel_mng(f->m_mpi_parallel_mng)
, m_receive_request_list(m_mpi_parallel_mng->createRequestListRef())
, m_send_request_list(m_mpi_parallel_mng->createRequestListRef())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiVariableSynchronizeDispatcher::
beginSynchronize(IDataSynchronizeBuffer* ds_buf)
{
  Integer nb_message = ds_buf->nbRank();

  m_send_request_list->clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;

  MP::Mpi::MpiAdapter* mpi_adapter = pm->adapter();
  const MPI_Datatype mpi_dt = MP::Mpi::MpiBuiltIn::datatype(Byte());

  double prepare_time = 0.0;

  {
    MpiTimeInterval tit(&prepare_time);
    constexpr int serialize_tag = 523;

    // Envoie les messages de réception en mode non bloquant
    m_original_recv_requests_done.resize(nb_message);
    m_original_recv_requests.resize(nb_message);

    // Poste les messages de réception
    for (Integer i = 0; i < nb_message; ++i) {
      Int32 target_rank = ds_buf->targetRank(i);
      auto buf = ds_buf->receiveBuffer(i).bytes();
      if (!buf.empty()) {
        auto req = mpi_adapter->receiveNonBlockingNoStat(buf.data(), buf.size(),
                                                         target_rank, mpi_dt, serialize_tag);
        m_original_recv_requests[i] = req;
        m_original_recv_requests_done[i] = false;
      }
      else {
        // Il n'est pas nécessaire d'envoyer un message vide.
        // Considère le message comme terminé
        m_original_recv_requests[i] = Parallel::Request{};
        m_original_recv_requests_done[i] = true;
      }
    }

    // Recopie les buffers d'envoi dans \a var_values
    ds_buf->copyAllSend();

    // Poste les messages d'envoi en mode non bloquant.
    for (Integer i = 0; i < nb_message; ++i) {
      auto buf = ds_buf->sendBuffer(i).bytes();
      Int32 target_rank = ds_buf->targetRank(i);
      if (!buf.empty()) {
        auto request = mpi_adapter->sendNonBlockingNoStat(buf.data(), buf.size(),
                                                          target_rank, mpi_dt, serialize_tag);
        m_send_request_list->add(request);
      }
    }
  }
  pm->stat()->add("SyncPrepare", prepare_time, ds_buf->totalSendSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiVariableSynchronizeDispatcher::
endSynchronize(IDataSynchronizeBuffer* ds_buf)
{
  MpiParallelMng* pm = m_mpi_parallel_mng;

  // On a besoin de conserver l'indice d'origine dans 'SyncBuffer'
  // de chaque requête pour gérer les copies.
  UniqueArray<Integer> remaining_original_indexes;

  double copy_time = 0.0;
  double wait_time = 0.0;

  while (1) {
    // Créé la liste des requêtes encore active.
    m_receive_request_list->clear();
    remaining_original_indexes.clear();
    for (Integer i = 0, n = m_original_recv_requests_done.size(); i < n; ++i) {
      if (!m_original_recv_requests_done[i]) {
        m_receive_request_list->add(m_original_recv_requests[i]);
        remaining_original_indexes.add(i);
      }
    }
    Integer nb_remaining_request = m_receive_request_list->size();
    if (nb_remaining_request == 0)
      break;

    {
      MpiTimeInterval tit(&wait_time);
      m_receive_request_list->wait(Parallel::WaitSome);
    }

    // Pour chaque requête terminée, effectue la copie
    ConstArrayView<Int32> done_requests = m_receive_request_list->doneRequestIndexes();

    for (Int32 request_index : done_requests) {
      Int32 orig_index = remaining_original_indexes[request_index];

      // Pour indiquer que c'est fini
      m_original_recv_requests_done[orig_index] = true;

      // Recopie les valeurs recues
      {
        MpiTimeInterval tit(&copy_time);
        ds_buf->copyReceiveAsync(orig_index);
      }
    }
  }

  // Attend que les envois se terminent.
  // Il faut le faire pour pouvoir libérer les requêtes même si le message
  // est arrivé.
  {
    MpiTimeInterval tit(&wait_time);
    m_send_request_list->wait(Parallel::WaitAll);
  }

  // S'assure que les copies des buffers sont bien terminées
  ds_buf->barrier();

  Int64 total_ghost_size = ds_buf->totalReceiveSize();
  Int64 total_share_size = ds_buf->totalSendSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy", copy_time, total_ghost_size);
  pm->stat()->add("SyncWait", wait_time, total_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
