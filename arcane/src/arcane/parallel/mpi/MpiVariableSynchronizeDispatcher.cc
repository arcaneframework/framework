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

  m_send_requests.clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MPI_Comm comm = pm->communicator();
  MpiDatatypeList* dtlist = pm->datatypes();

  //ITraceMng* trace = pm->traceMng();
  //trace->info() << " ** ** MPI BEGIN SYNC n=" << nb_message
  //              << " this=" << (IVariableSynchronizeDispatcher*)this;
  //trace->flush();

  //SyncBuffer& sync_buffer = this->m_1d_buffer;
  // Envoie les messages de réception en mode non bloquant
  m_recv_requests.resize(nb_message);
  m_recv_requests_done.resize(nb_message);
  double begin_prepare_time = MPI_Wtime();
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = this->m_sync_list[i];
      ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[i];
      if (!ghost_local_buffer.empty()){
        MPI_Request mpi_request;
        MPI_Datatype dt = dtlist->datatype(SimpleType())->datatype();
        m_mpi_parallel_mng->adapter()->getMpiProfiling()->iRecv(ghost_local_buffer.data(),ghost_local_buffer.size(),
                                                                dt,vsi.m_target_rank,523,comm,&mpi_request);
        
        m_recv_requests[i] = mpi_request;
        m_recv_requests_done[i] = false;
        //trace->info() << "POST RECV " << vsi.m_target_rank;
      }
      else{
        // Il n'est pas nécessaire d'envoyer un message vide.
        // Considère le message comme terminé
        m_recv_requests[i] = MPI_Request();
        m_recv_requests_done[i] = true;
      }
    }

    // Envoie les messages d'envoie en mode non bloquant.
    for( Integer i=0; i<nb_message; ++i ){
      const VariableSyncInfo& vsi = this->m_sync_list[i];
      Int32ConstArrayView share_grp = vsi.m_share_ids;
      ArrayView<SimpleType> share_local_buffer = sync_buffer.m_share_locals_buffer[i];
      this->_copyToBuffer(share_grp,share_local_buffer,var_values,dim2_size);
      if (!share_local_buffer.empty()){
        MPI_Request mpi_request;
        MPI_Datatype dt = dtlist->datatype(SimpleType())->datatype();
        m_mpi_parallel_mng->adapter()->getMpiProfiling()->iSend(share_local_buffer.data(),share_local_buffer.size(),
                                                                dt,vsi.m_target_rank,523,comm,&mpi_request);
        m_send_requests.add(mpi_request);
        //trace->info() << "POST SEND " << vsi.m_target_rank;
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
endSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (!this->m_is_in_sync)
    ARCANE_FATAL("endSynchronize() called but no beginSynchronize() was called before");

  Integer dim2_size = sync_buffer.m_dim2_size;

  MpiParallelMng* pm = m_mpi_parallel_mng;

  //ITraceMng* trace = pm->traceMng();
  //trace->info() << " ** ** MPI END SYNC "
  //              << " this=" << (IVariableSynchronizeDispatcher*)this;

  UniqueArray<MPI_Request> remaining_request;
  UniqueArray<Integer> remaining_indexes;

  UniqueArray<MPI_Status> mpi_status;
  UniqueArray<int> completed_requests;

  UniqueArray<MPI_Request> m_remaining_recv_requests;
  UniqueArray<Integer> m_remaining_recv_request_indexes;
  double copy_time = 0.0;
  double wait_time = 0.0;
  while(1){
    m_remaining_recv_requests.clear();
    m_remaining_recv_request_indexes.clear();
    for( Integer i=0; i<m_recv_requests.size(); ++i ){
      if (!m_recv_requests_done[i]){
        m_remaining_recv_requests.add(m_recv_requests[i]);
        m_remaining_recv_request_indexes.add(i); //m_recv_request_indexes[i]);
      }
    }
    Integer nb_remaining_request = m_remaining_recv_requests.size();
    if (nb_remaining_request==0)
      break;
    int nb_completed_request = 0;
    mpi_status.resize(nb_remaining_request);
    completed_requests.resize(nb_remaining_request);
    {
      double begin_time = MPI_Wtime();
      //trace->info() << "Wait some: n=" << nb_remaining_request
      //              << " total=" << nb_message;
      m_mpi_parallel_mng->adapter()->getMpiProfiling()->waitSome(nb_remaining_request,m_remaining_recv_requests.data(),
                                                                 &nb_completed_request,completed_requests.data(),
                                                                 mpi_status.data());
      //trace->info() << "Wait some end: nb_done=" << nb_completed_request;
      double end_time = MPI_Wtime();
      wait_time += (end_time-begin_time);
    }
    // Pour chaque requete terminee, effectue la copie
    for( int z=0; z<nb_completed_request; ++z ){
      int mpi_request_index = completed_requests[z];
      Integer index = m_remaining_recv_request_indexes[mpi_request_index];

      {
        double begin_time = MPI_Wtime();
        const VariableSyncInfo& vsi = this->m_sync_list[index];
        Int32ConstArrayView ghost_grp = vsi.m_ghost_ids;
        ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[index];
        this->_copyFromBuffer(ghost_grp,ghost_local_buffer,var_values,dim2_size);
        double end_time = MPI_Wtime();
        copy_time += (end_time - begin_time);
      }
      //trace->info() << "Mark finish index = " << index << " mpi_request_index=" << mpi_request_index;
      m_recv_requests_done[index] = true; // Pour indiquer que c'est fini
    }
  }

  //trace->info() << "Wait all begin: n=" << m_send_requests.size();
  // Attend que les envois se terminent
  mpi_status.resize(m_send_requests.size());
  m_mpi_parallel_mng->adapter()->getMpiProfiling()->waitAll(m_send_requests.size(),m_send_requests.data(),
                                                            mpi_status.data());
  //trace->info() << "Wait all end";

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
