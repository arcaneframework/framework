// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/parallel/IStat.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arccore/message_passing/IRequestList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: Séparer le cas avec type dérivé dans une classe à part.
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
compute(ConstArrayView<VariableSyncInfo> sync_list)
{
  //m_mpi_parallel_mng->traceMng()->info() << "MPI COMPUTE";
  VariableSynchronizeDispatcher<SimpleType>::compute(sync_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
beginSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (this->m_is_in_sync)
    ARCANE_FATAL("Only one pending serialisation is supported");

  Integer nb_message = this->m_sync_list.size();
  Integer dim2_size = sync_buffer.m_dim2_size;

  m_send_request_list->clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();

  //ITraceMng* trace = pm->traceMng();
  //trace->info() << " ** ** MPI BEGIN SYNC n=" << nb_message
  //              << " this=" << (IVariableSynchronizeDispatcher*)this;
  //trace->flush();

  MP::Mpi::MpiAdapter* mpi_adapter = m_mpi_parallel_mng->adapter();

  constexpr int serialize_tag = 523;
  // Envoie les messages de réception en mode non bloquant
  m_original_recv_requests_done.resize(nb_message);
  m_original_recv_requests.resize(nb_message);
  double begin_prepare_time = MPI_Wtime();

  // Poste les messages de réception
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = this->m_sync_list[i];
    ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[i];
    if (!ghost_local_buffer.empty()){
      MPI_Datatype dt = dtlist->datatype(SimpleType())->datatype();
      auto req = mpi_adapter->directRecv(ghost_local_buffer.data(),ghost_local_buffer.size(),
                                         vsi.m_target_rank,sizeof(SimpleType),dt,serialize_tag,false);
      m_original_recv_requests[i] = req;
      m_original_recv_requests_done[i] = false;
      //trace->info() << "POST RECV " << vsi.m_target_rank;
    }
    else{
      // Il n'est pas nécessaire d'envoyer un message vide.
      // Considère le message comme terminé
      m_original_recv_requests[i] = Parallel::Request{};
      m_original_recv_requests_done[i] = true;
    }
  }

  // Envoie les messages d'envoie en mode non bloquant.
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = this->m_sync_list[i];
    Int32ConstArrayView share_grp = vsi.m_share_ids;
    ArrayView<SimpleType> share_local_buffer = sync_buffer.m_share_locals_buffer[i];
    this->_copyToBuffer(share_grp,share_local_buffer,var_values,dim2_size);
    if (!share_local_buffer.empty()){
      MPI_Datatype dt = dtlist->datatype(SimpleType())->datatype();
      auto request = mpi_adapter->directSend(share_local_buffer.data(),share_local_buffer.size(),
                                             vsi.m_target_rank,sizeof(SimpleType),dt,serialize_tag,false);
      m_send_request_list->add(request);
    }
  }
  double prepare_time = MPI_Wtime() - begin_prepare_time;
  pm->stat()->add("SyncPrepare",prepare_time,1);
  this->m_is_in_sync = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
_copyReceive(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer,Integer index)
{
  Integer dim2_size = sync_buffer.m_dim2_size;
  const VariableSyncInfo& vsi = this->m_sync_list[index];
  ConstArrayView<Int32> ghost_grp = vsi.m_ghost_ids;
  ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[index];
  this->_copyFromBuffer(ghost_grp,ghost_local_buffer,var_values,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
endSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (!this->m_is_in_sync)
    ARCANE_FATAL("endSynchronize() called but no beginSynchronize() was called before");

  MpiParallelMng* pm = m_mpi_parallel_mng;

  UniqueArray<Integer> remaining_receive_request_indexes;
  double copy_time = 0.0;
  double wait_time = 0.0;

  while(1){
    m_receive_request_list->clear();
    remaining_receive_request_indexes.clear();
    for( Integer i=0, n=m_original_recv_requests_done.size(); i<n; ++i ){
      if (!m_original_recv_requests_done[i]){
        m_receive_request_list->add(m_original_recv_requests[i]);
        remaining_receive_request_indexes.add(i);
      }
    }
    Integer nb_remaining_request = m_receive_request_list->size();
    if (nb_remaining_request==0)
      break;

    {
      double begin_time = MPI_Wtime();
      m_receive_request_list->wait(Parallel::WaitSome);
      double end_time = MPI_Wtime();
      wait_time += (end_time-begin_time);
    }

    // Pour chaque requete terminée, effectue la copie
    ConstArrayView<Int32> done_requests = m_receive_request_list->doneRequestIndexes();
    Integer nb_completed_request = done_requests.size();

    for( Integer z=0; z<nb_completed_request; ++z ){
      Integer mpi_request_index = done_requests[z];
      Integer index = remaining_receive_request_indexes[mpi_request_index];

      double begin_time = MPI_Wtime();
      _copyReceive(var_values,sync_buffer,index);
      double end_time = MPI_Wtime();
      copy_time += (end_time - begin_time);

      m_original_recv_requests_done[index] = true; // Pour indiquer que c'est fini
    }
  }

  // Attend que les envois se terminent.
  // Il faut le faire pour pouvoir libérer les requêtes même si le message
  // est arrivé.
  m_send_request_list->wait(Parallel::WaitAll);

  pm->stat()->add("SyncCopy",copy_time,1);
  pm->stat()->add("SyncWait",wait_time,1);
  this->m_is_in_sync = false;
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
