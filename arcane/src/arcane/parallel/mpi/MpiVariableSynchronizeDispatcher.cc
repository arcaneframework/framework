// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiVariableSynchronizeDispatcher.cc                         (C) 2000-2021 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/parallel/mpi/MpiVariableSynchronizeDispatcher.h"
#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiAdapter.h"
#include "arcane/parallel/mpi/MpiDatatypeList.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/MpiTimeInterval.h"
#include "arcane/parallel/IStat.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arccore/message_passing/IRequestList.h"

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

template<typename SimpleType>
MpiVariableSynchronizeDispatcher<SimpleType>::
MpiVariableSynchronizeDispatcher(MpiVariableSynchronizeDispatcherBuildInfo& bi)
: VariableSynchronizeDispatcher<SimpleType>(VariableSynchronizeDispatcherBuildInfo(bi.parallelMng(),bi.table()))
, m_mpi_parallel_mng(bi.parallelMng())
, m_receive_request_list(m_mpi_parallel_mng->createRequestListRef())
, m_send_request_list(m_mpi_parallel_mng->createRequestListRef())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
compute(ItemGroupSynchronizeInfo* sync_info)
{
  VariableSynchronizeDispatcher<SimpleType>::compute(sync_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
_beginSynchronize(SyncBuffer& sync_buffer)
{
  auto sync_list = this->m_sync_info->infos();
  Integer nb_message = sync_list.size();

  m_send_request_list->clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();

  MP::Mpi::MpiAdapter* mpi_adapter = pm->adapter();

  double prepare_time = 0.0;

  {
    MpiTimeInterval tit(&prepare_time);
    constexpr int serialize_tag = 523;
    const MPI_Datatype mpi_dt = dtlist->datatype(SimpleType())->datatype();

    // Envoie les messages de réception en mode non bloquant
    m_original_recv_requests_done.resize(nb_message);
    m_original_recv_requests.resize(nb_message);

    // Poste les messages de réception
    for (Integer i = 0; i < nb_message; ++i) {
      const VariableSyncInfo& vsi = sync_list[i];
      ArrayView<SimpleType> buf = sync_buffer.ghostBuffer(i);
      if (!buf.empty()) {
        auto req = mpi_adapter->receiveNonBlockingNoStat(buf.data(), buf.size(),
                                                         vsi.targetRank(), mpi_dt, serialize_tag);
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
    for (Integer i = 0; i < nb_message; ++i)
      sync_buffer.copySend(i);

    // Poste les messages d'envoi en mode non bloquant.
    for (Integer i = 0; i < nb_message; ++i) {
      ArrayView<SimpleType> buf = sync_buffer.shareBuffer(i);
      const VariableSyncInfo& vsi = sync_list[i];
      if (!buf.empty()) {
        auto request = mpi_adapter->sendNonBlockingNoStat(buf.data(), buf.size(),
                                                          vsi.targetRank(), mpi_dt, serialize_tag);
        m_send_request_list->add(request);
      }
    }
  }
  pm->stat()->add("SyncPrepare", prepare_time, sync_buffer.totalShareSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBuffer& sync_buffer)
{
  MpiParallelMng* pm = m_mpi_parallel_mng;

  // On a besoin de conserver l'indice d'origine dans 'SyncBuffer'
  // de chaque requête pour gérer les copies.
  UniqueArray<Integer> remaining_original_indexes;

  double copy_time = 0.0;
  double wait_time = 0.0;

  while(1){
    // Créé la liste des requêtes encore active.
    m_receive_request_list->clear();
    remaining_original_indexes.clear();
    for( Integer i=0, n=m_original_recv_requests_done.size(); i<n; ++i ){
      if (!m_original_recv_requests_done[i]){
        m_receive_request_list->add(m_original_recv_requests[i]);
        remaining_original_indexes.add(i);
      }
    }
    Integer nb_remaining_request = m_receive_request_list->size();
    if (nb_remaining_request==0)
      break;

    {
      MpiTimeInterval tit(&wait_time);
      m_receive_request_list->wait(Parallel::WaitSome);
    }

    // Pour chaque requête terminée, effectue la copie
    ConstArrayView<Int32> done_requests = m_receive_request_list->doneRequestIndexes();

    for( Int32 request_index : done_requests ){
      Int32 orig_index = remaining_original_indexes[request_index];

      // Pour indiquer que c'est fini
      m_original_recv_requests_done[orig_index] = true;

      // Recopie les valeurs recues
      {
        MpiTimeInterval tit(&copy_time);
        sync_buffer.copyReceive(orig_index);
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

  Int64 total_ghost_size = sync_buffer.totalGhostSize();
  Int64 total_share_size = sync_buffer.totalShareSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy",copy_time,total_ghost_size);
  pm->stat()->add("SyncWait",wait_time,total_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiVariableSynchronizeDispatcher<Byte>;
template class MpiVariableSynchronizeDispatcher<Int16>;
template class MpiVariableSynchronizeDispatcher<Int32>;
template class MpiVariableSynchronizeDispatcher<Int64>;
template class MpiVariableSynchronizeDispatcher<Real>;
template class MpiVariableSynchronizeDispatcher<Real2>;
template class MpiVariableSynchronizeDispatcher<Real3>;
template class MpiVariableSynchronizeDispatcher<Real2x2>;
template class MpiVariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
