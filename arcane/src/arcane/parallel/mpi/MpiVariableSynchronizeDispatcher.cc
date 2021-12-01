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

namespace
{
class TimeInterval
{
 public:
  TimeInterval(double* cumulative_value)
  : m_cumulative_value(cumulative_value)
  {
    m_begin_time = MPI_Wtime();
  }
  ~TimeInterval()
  {
    double end_time = MPI_Wtime();
    *m_cumulative_value = (end_time - m_begin_time);
  }
 private:
  double* m_cumulative_value;
  double m_begin_time = 0.0;
};

}

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
compute(ItemGroupSynchronizeInfo* sync_info)
{
  VariableSynchronizeDispatcher<SimpleType>::compute(sync_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
_beginSynchronize(SyncBuffer& sync_buffer)
{
  auto sync_list = this->m_sync_info->infos();
  Integer nb_message = sync_list.size();

  m_send_request_list->clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();

  MP::Mpi::MpiAdapter* mpi_adapter = m_mpi_parallel_mng->adapter();

  double begin_prepare_time = MPI_Wtime();

  constexpr int serialize_tag = 523;
  const MPI_Datatype mpi_dt = dtlist->datatype(SimpleType())->datatype();

  // Envoie les messages de réception en mode non bloquant
  m_original_recv_requests_done.resize(nb_message);
  m_original_recv_requests.resize(nb_message);

  // Poste les messages de réception
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = sync_list[i];
    ArrayView<SimpleType> buf = sync_buffer.ghostBuffer(i);
    if (!buf.empty()){
      auto req = mpi_adapter->receiveNonBlockingNoStat(buf.data(),buf.size(),
                                                       vsi.targetRank(),mpi_dt,serialize_tag);
      m_original_recv_requests[i] = req;
      m_original_recv_requests_done[i] = false;
    }
    else{
      // Il n'est pas nécessaire d'envoyer un message vide.
      // Considère le message comme terminé
      m_original_recv_requests[i] = Parallel::Request{};
      m_original_recv_requests_done[i] = true;
    }
  }

  // Recopie les buffers d'envoi dans \a var_values
  for( Integer i=0; i<nb_message; ++i )
    sync_buffer.copySend(i);

  // Poste les messages d'envoie en mode non bloquant.
  for( Integer i=0; i<nb_message; ++i ){
    ArrayView<SimpleType> buf = sync_buffer.shareBuffer(i);
    const VariableSyncInfo& vsi = sync_list[i];
    if (!buf.empty()){
      auto request = mpi_adapter->sendNonBlockingNoStat(buf.data(),buf.size(),
                                                        vsi.targetRank(),mpi_dt,serialize_tag);
      m_send_request_list->add(request);
    }
  }
  double prepare_time = MPI_Wtime() - begin_prepare_time;
  pm->stat()->add("SyncPrepare",prepare_time,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBuffer& sync_buffer)
{
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
      TimeInterval tit(&wait_time);
      m_receive_request_list->wait(Parallel::WaitSome);
    }

    // Pour chaque requete terminée, effectue la copie
    ConstArrayView<Int32> done_requests = m_receive_request_list->doneRequestIndexes();
    Integer nb_completed_request = done_requests.size();

    for( Integer z=0; z<nb_completed_request; ++z ){
      Integer mpi_request_index = done_requests[z];
      Integer index = remaining_receive_request_indexes[mpi_request_index];
      m_original_recv_requests_done[index] = true; // Pour indiquer que c'est fini

      // Recopie les valeurs recues
      {
        TimeInterval tit(&copy_time);
        sync_buffer.copyReceive(index);
      }
    }
  }

  // Attend que les envois se terminent.
  // Il faut le faire pour pouvoir libérer les requêtes même si le message
  // est arrivé.
  {
    TimeInterval tit(&wait_time);
    m_send_request_list->wait(Parallel::WaitAll);
  }

  pm->stat()->add("SyncCopy",copy_time,1);
  pm->stat()->add("SyncWait",wait_time,1);
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
