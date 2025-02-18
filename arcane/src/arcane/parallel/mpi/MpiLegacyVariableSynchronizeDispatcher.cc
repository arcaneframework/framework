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

#include "arcane/utils/MemoryView.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiDatatypeList.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/IStat.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/impl/IDataSynchronizeImplementation.h"

#include "arccore/message_passing_mpi/internal/IMpiProfiling.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation optimisée pour MPI de la synchronisation.
 *
 * Cette classe implémente la version historique de la synchronisation qui existe
 * dans les versions de Arcane antérieures à la 3.2.
 *
 * Par rapport à la version de base, cette implémentation fait un MPI_Waitsome
 * (au lieu d'un Waitall) et recopie dans le buffer de destination
 * dès qu'un message arrive.
 */
class MpiLegacyVariableSynchronizerDispatcher
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit MpiLegacyVariableSynchronizerDispatcher(Factory* f);

 protected:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* buf) override;
  void endSynchronize(IDataSynchronizeBuffer* buf) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng;
  UniqueArray<MPI_Request> m_send_requests;
  UniqueArray<MPI_Request> m_recv_requests;
  UniqueArray<Integer> m_recv_requests_done;
  UniqueArray<MPI_Datatype> m_share_derived_types;
  UniqueArray<MPI_Datatype> m_ghost_derived_types;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiLegacyVariableSynchronizerDispatcher::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  explicit Factory(MpiParallelMng* mpi_pm)
  : m_mpi_parallel_mng(mpi_pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new MpiLegacyVariableSynchronizerDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiLegacyVariableSynchronizerFactory(MpiParallelMng* mpi_pm)
{
  auto* x = new MpiLegacyVariableSynchronizerDispatcher::Factory(mpi_pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiLegacyVariableSynchronizerDispatcher::
MpiLegacyVariableSynchronizerDispatcher(Factory* f)
: m_mpi_parallel_mng(f->m_mpi_parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiLegacyVariableSynchronizerDispatcher::
beginSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  Integer nb_message = vs_buf->nbRank();

  m_send_requests.clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MPI_Comm comm = pm->communicator();
  MpiDatatypeList* dtlist = pm->datatypes();
  MP::Mpi::IMpiProfiling* mpi_profiling = m_mpi_parallel_mng->adapter()->getMpiProfiling();

  //ITraceMng* trace = pm->traceMng();
  //trace->info() << " ** ** MPI BEGIN SYNC n=" << nb_message
  //              << " this=" << (IVariableSynchronizeDispatcher*)this;
  //trace->flush();

  MPI_Datatype byte_dt = dtlist->datatype(Byte())->datatype();

  //SyncBuffer& sync_buffer = this->m_1d_buffer;
  // Envoie les messages de réception en mode non bloquant
  m_recv_requests.resize(nb_message);
  m_recv_requests_done.resize(nb_message);
  double begin_prepare_time = MPI_Wtime();
  for( Integer i=0; i<nb_message; ++i ){
    Int32 target_rank = vs_buf->targetRank(i);
    auto ghost_local_buffer = vs_buf->receiveBuffer(i).bytes().smallView();
    if (!ghost_local_buffer.empty()){
      MPI_Request mpi_request;
      mpi_profiling->iRecv(ghost_local_buffer.data(),ghost_local_buffer.size(),
                           byte_dt,target_rank,523,comm,&mpi_request);
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

  vs_buf->copyAllSend();

  // Envoie les messages d'envoi en mode non bloquant.
  for( Integer i=0; i<nb_message; ++i ){
    Int32 target_rank = vs_buf->targetRank(i);
    auto share_local_buffer = vs_buf->sendBuffer(i).bytes().smallView();
    if (!share_local_buffer.empty()){
      MPI_Request mpi_request;
      mpi_profiling->iSend(share_local_buffer.data(),share_local_buffer.size(),
                           byte_dt,target_rank,523,comm,&mpi_request);
      m_send_requests.add(mpi_request);
    }
  }
  {
    double prepare_time = MPI_Wtime() - begin_prepare_time;
    pm->stat()->add("SyncPrepare",prepare_time,vs_buf->totalSendSize());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiLegacyVariableSynchronizerDispatcher::
endSynchronize(IDataSynchronizeBuffer* vs_buf)
{
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
    // Pour chaque requête terminée, effectue la copie
    for( int z=0; z<nb_completed_request; ++z ){
      int mpi_request_index = completed_requests[z];
      Integer index = m_remaining_recv_request_indexes[mpi_request_index];

      {
        double begin_time = MPI_Wtime();
        vs_buf->copyReceiveAsync(index);
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

  // S'assure que les copies des buffers sont bien terminées
  vs_buf->barrier();

  //trace->info() << "Wait all end";
  {
    Int64 total_ghost_size = vs_buf->totalReceiveSize();
    Int64 total_share_size = vs_buf->totalSendSize();
    Int64 total_size = total_ghost_size + total_share_size;
    pm->stat()->add("SyncCopy",copy_time,total_ghost_size);
    pm->stat()->add("SyncWait",wait_time,total_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
