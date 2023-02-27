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

#include "arcane/parallel/mpi/MpiLegacyVariableSynchronizeDispatcher.h"
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
MpiLegacyVariableSynchronizeDispatcher<SimpleType>::
MpiLegacyVariableSynchronizeDispatcher(MpiLegacyVariableSynchronizeDispatcherBuildInfo& bi)
: VariableSynchronizeDispatcher<SimpleType>(VariableSynchronizeDispatcherBuildInfo(bi.parallelMng(),bi.table()))
, m_mpi_parallel_mng(bi.parallelMng())
, m_use_derived_type(true)
{
  //NOTE: Desactive pour l'instant les types derives car cela ne fonctionne
  // par correctement avec BullMPI.
  // Verifier si cela vient de Arcane ou de Bull.
  // Ca fonctionne correctement avec Mpich2 1.0.5 et OpenMpi 1.2+
  m_use_derived_type = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiLegacyVariableSynchronizeDispatcher<SimpleType>::
compute()
{
  //m_mpi_parallel_mng->traceMng()->info() << "MPI COMPUTE";
  VariableSynchronizeDispatcher<SimpleType>::compute();
  auto sync_list = this->m_sync_info->infos();
  if (m_use_derived_type){
    //TODO Utiliser des 'int' MPI au lieu de Int32
    Integer nb_message = sync_list.size();
    MpiParallelMng* pm = m_mpi_parallel_mng;
    _destroyTypes();
    m_share_derived_types.resize(nb_message);
    m_ghost_derived_types.resize(nb_message);
    //TODO Utiliser des 'int' MPI au lieu de Int32
    pm->traceMng()->info() << "CREATE DERIVED TYPE";
    typedef DataTypeTraitsT<SimpleType> DataTypeTraits;
    typedef typename DataTypeTraitsT<SimpleType>::BasicType BasicType;
    int nb_basic = DataTypeTraits::nbBasicType();
    MPI_Datatype mpi_basetype = pm->datatypes()->datatype(BasicType())->datatype();
    UniqueArray<int> ids;
    for( Integer i=0; i<nb_message; ++i ){
      const VariableSyncInfo& vsi = sync_list[i];
      Int32ConstArrayView share_grp = vsi.shareIds();
      Integer nb_share = share_grp.size();
      ids.resize(nb_share);
      for( Integer z=0; z<nb_share; ++z )
        ids[z] = share_grp[z]*nb_basic;
      MPI_Type_create_indexed_block(ids.size(),nb_basic,ids.data(),
                                    mpi_basetype,&m_share_derived_types[i]);
      MPI_Type_commit(&m_share_derived_types[i]);

      Int32ConstArrayView ghost_grp = vsi.ghostIds();
      Integer nb_ghost = ghost_grp.size();
      ids.resize(nb_ghost);
      for( Integer z=0; z<nb_ghost; ++z )
        ids[z] = ghost_grp[z]*nb_basic;
      MPI_Type_create_indexed_block(ids.size(),nb_basic,ids.data(),
                                    mpi_basetype,&m_ghost_derived_types[i]);
      MPI_Type_commit(&m_ghost_derived_types[i]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiLegacyVariableSynchronizeDispatcher<SimpleType>::
_beginSynchronize(SyncBuffer& sync_buffer)
{
  MutableMemoryView var_values = sync_buffer.dataMemoryView();
  auto sync_list = this->m_sync_info->infos();
  //Integer nb_elem = var_values.size();
  Integer nb_message = sync_list.size();
  Integer dim2_size = sync_buffer.dim2Size();

  m_send_requests.clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MPI_Comm comm = pm->communicator();
  MpiDatatypeList* dtlist = pm->datatypes();
  MP::Mpi::IMpiProfiling* mpi_profiling = m_mpi_parallel_mng->adapter()->getMpiProfiling();

  //ITraceMng* trace = pm->traceMng();
  //trace->info() << " ** ** MPI BEGIN SYNC n=" << nb_message
  //              << " this=" << (IVariableSynchronizeDispatcher*)this;
  //trace->flush();

  bool use_derived = (dim2_size==1 && m_use_derived_type);

  //SyncBuffer& sync_buffer = this->m_1d_buffer;
  // Envoie les messages de réception en mode non bloquant
  m_recv_requests.resize(nb_message);
  m_recv_requests_done.resize(nb_message);
  double begin_prepare_time = MPI_Wtime();
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = sync_list[i];
    ArrayView<SimpleType> ghost_local_buffer = sync_buffer.ghostBuffer(i);
      if (!ghost_local_buffer.empty()){
        MPI_Request mpi_request;
        if (use_derived){
          mpi_profiling->iRecv(var_values.data(),1,m_ghost_derived_types[i],
                               vsi.targetRank(),523,comm,&mpi_request);
        }
        else{
          MPI_Datatype dt = dtlist->datatype(SimpleType())->datatype();
          mpi_profiling->iRecv(ghost_local_buffer.data(),ghost_local_buffer.size(),
                               dt,vsi.targetRank(),523,comm,&mpi_request);
        }
        
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
      ArrayView<SimpleType> share_local_buffer = sync_buffer.shareBuffer(i);
      if (!use_derived)
        sync_buffer.copySend(i);
      if (!share_local_buffer.empty()){
        MPI_Request mpi_request;
        if (use_derived){
          mpi_profiling->iSend(var_values.data(),1,m_share_derived_types[i],
                               vsi.targetRank(),523,comm,&mpi_request);
        }
        else{
          MPI_Datatype dt = dtlist->datatype(SimpleType())->datatype();
          mpi_profiling->iSend(share_local_buffer.data(),share_local_buffer.size(),
                               dt,vsi.targetRank(),523,comm,&mpi_request);
        }
        m_send_requests.add(mpi_request);
        //trace->info() << "POST SEND " << vsi.m_target_rank;
      }
    }
    double prepare_time = MPI_Wtime() - begin_prepare_time;
    if (use_derived){
      pm->stat()->add("SyncPrepareDerived",prepare_time,1);
    }
    else{
      pm->stat()->add("SyncPrepare",prepare_time,sync_buffer.totalShareSize());
    }
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiLegacyVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBuffer& sync_buffer)
{
  Integer dim2_size = sync_buffer.dim2Size();
  bool use_derived = (dim2_size==1 && m_use_derived_type);

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

      if (!use_derived){
        double begin_time = MPI_Wtime();
        sync_buffer.copyReceive(index);
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

  Int64 total_ghost_size = sync_buffer.totalGhostSize();
  Int64 total_share_size = sync_buffer.totalShareSize();
  Int64 total_size = total_ghost_size + total_share_size;
  if (use_derived){
    pm->stat()->add("SyncWaitDerived",wait_time,total_size);
  }
  else{
    pm->stat()->add("SyncCopy",copy_time,total_ghost_size);
    pm->stat()->add("SyncWait",wait_time,total_size);
  }
  this->m_is_in_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiLegacyVariableSynchronizeDispatcher<Byte>;
template class MpiLegacyVariableSynchronizeDispatcher<Int16>;
template class MpiLegacyVariableSynchronizeDispatcher<Int32>;
template class MpiLegacyVariableSynchronizeDispatcher<Int64>;
template class MpiLegacyVariableSynchronizeDispatcher<Real>;
template class MpiLegacyVariableSynchronizeDispatcher<Real2>;
template class MpiLegacyVariableSynchronizeDispatcher<Real3>;
template class MpiLegacyVariableSynchronizeDispatcher<Real2x2>;
template class MpiLegacyVariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
