// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiBlockVariableSynchronizeDispatcher.cc                    (C) 2000-2025 */
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
 * Cette implémentation découpe la synchronisation en bloc de taille fixe.
 * Tout le mécanisme est dans _endSynchronize().
 * L'algorithme est le suivant:
 *
 * 1. Recopie dans les buffers d'envoi les valeurs à envoyer.
 * 2. Boucle sur Irecv/ISend/WaitAll tant que qu'il y a au moins une partie non vide.
 * 3. Recopie depuis les buffers de réception les valeurs des variables.
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation par block pour MPI de la synchronisation.
 *
 * Les messages sont envoyés par bloc d'une taille fixe.
 *
 * NOTE: cette optimisation respecte la norme MPI qui dit qu'on ne doit
 * plus toucher à la zone mémoire d'un message tant que celui-ci n'est
 * pas fini.
 */
class MpiBlockVariableSynchronizerDispatcher
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit MpiBlockVariableSynchronizerDispatcher(Factory* f);

 public:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* buf) override;
  void endSynchronize(IDataSynchronizeBuffer* buf) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  Ref<Parallel::IRequestList> m_request_list;
  Int32 m_block_size;
  Int32 m_nb_sequence;

 private:

  bool _isSkipRank(Int32 rank, Int32 sequence) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiBlockVariableSynchronizerDispatcher::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  Factory(MpiParallelMng* mpi_pm, Int32 block_size, Int32 nb_sequence)
  : m_mpi_parallel_mng(mpi_pm)
  , m_block_size(block_size)
  , m_nb_sequence(nb_sequence)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new MpiBlockVariableSynchronizerDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  Int32 m_block_size = 0;
  Int32 m_nb_sequence = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiBlockVariableSynchronizerFactory(MpiParallelMng* mpi_pm, Int32 block_size, Int32 nb_sequence)
{
  auto* x = new MpiBlockVariableSynchronizerDispatcher::Factory(mpi_pm, block_size, nb_sequence);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiBlockVariableSynchronizerDispatcher::
MpiBlockVariableSynchronizerDispatcher(Factory* f)
: m_mpi_parallel_mng(f->m_mpi_parallel_mng)
, m_request_list(m_mpi_parallel_mng->createRequestListRef())
, m_block_size(f->m_block_size)
, m_nb_sequence(f->m_nb_sequence)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiBlockVariableSynchronizerDispatcher::
_isSkipRank(Int32 rank, Int32 sequence) const
{
  if (m_nb_sequence == 1)
    return false;
  return (rank % m_nb_sequence) == sequence;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiBlockVariableSynchronizerDispatcher::
beginSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  // Ne fait rien au niveau MPI dans cette partie car cette implémentation
  // ne supporte pas l'asyncrhonisme.
  // On se contente de recopier les valeurs des variables dans le buffer d'envoi
  // pour permettre ensuite de modifier les valeurs de la variable entre
  // le _beginSynchronize() et le _endSynchronize().

  double send_copy_time = 0.0;
  {
    MpiTimeInterval tit(&send_copy_time);
    // Recopie les buffers d'envoi
    vs_buf->copyAllSend();
  }
  Int64 total_share_size = vs_buf->totalSendSize();
  m_mpi_parallel_mng->stat()->add("SyncSendCopy", send_copy_time, total_share_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiBlockVariableSynchronizerDispatcher::
endSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  const Int32 nb_message = vs_buf->nbRank();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  Int32 my_rank = pm->commRank();

  MP::Mpi::MpiAdapter* mpi_adapter = pm->adapter();
  const MPI_Datatype mpi_dt = MP::Mpi::MpiBuiltIn::datatype(Byte());

  double prepare_time = 0.0;
  double copy_time = 0.0;
  double wait_time = 0.0;

  constexpr int serialize_tag = 523;

  const Int32 block_size = m_block_size;

  for (Int32 isequence = 0; isequence < m_nb_sequence; ++isequence) {
    Int32 block_index = 0;
    while (1) {
      {
        MpiTimeInterval tit(&prepare_time);
        m_request_list->clear();

        // Poste les messages de réception
        for (Integer i = 0; i < nb_message; ++i) {
          Int32 target_rank = vs_buf->targetRank(i);
          if (_isSkipRank(target_rank, isequence))
            continue;
          auto buf0 = vs_buf->receiveBuffer(i).bytes();
          auto buf = buf0.subSpan(block_index, block_size);
          if (!buf.empty()) {
            auto req = mpi_adapter->receiveNonBlockingNoStat(buf.data(), buf.size(),
                                                             target_rank, mpi_dt, serialize_tag);
            m_request_list->add(req);
          }
        }

        // Poste les messages d'envoi en mode non bloquant.
        for (Integer i = 0; i < nb_message; ++i) {
          Int32 target_rank = vs_buf->targetRank(i);
          if (_isSkipRank(my_rank, isequence))
            continue;
          auto buf0 = vs_buf->sendBuffer(i).bytes();
          auto buf = buf0.subSpan(block_index, block_size);
          if (!buf.empty()) {
            auto request = mpi_adapter->sendNonBlockingNoStat(buf.data(), buf.size(),
                                                              target_rank, mpi_dt, serialize_tag);
            m_request_list->add(request);
          }
        }
      }

      // Si aucune requête alors on a fini notre synchronisation
      if (m_request_list->size() == 0)
        break;

      // Attend que les messages soient terminés
      {
        MpiTimeInterval tit(&wait_time);
        m_request_list->wait(Parallel::WaitAll);
      }

      block_index += block_size;
    }
  }

  // Recopie les valeurs recues
  {
    MpiTimeInterval tit(&copy_time);
    vs_buf->copyAllReceive();
  }

  Int64 total_ghost_size = vs_buf->totalReceiveSize();
  Int64 total_share_size = vs_buf->totalSendSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy", copy_time, total_ghost_size);
  pm->stat()->add("SyncWait", wait_time, total_size);
  pm->stat()->add("SyncPrepare", prepare_time, total_share_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
