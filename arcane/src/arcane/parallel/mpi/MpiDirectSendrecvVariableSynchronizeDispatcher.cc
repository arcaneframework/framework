// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDirectSendrecvVariableSynchronizeDispatcher.cc           (C) 2000-2022 */
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

#include "arcane/parallel/mpi/MpiDirectSendrecvVariableSynchronizeDispatcher.h"
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


namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType>
MpiDirectSendrecvVariableSynchronizeDispatcher<SimpleType>::
MpiDirectSendrecvVariableSynchronizeDispatcher(MpiDirectSendrecvVariableSynchronizeDispatcherBuildInfo& bi)
: VariableSynchronizeDispatcher<SimpleType>(VariableSynchronizeDispatcherBuildInfo(bi.parallelMng(),bi.table()))
, m_mpi_parallel_mng(bi.parallelMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiDirectSendrecvVariableSynchronizeDispatcher<SimpleType>::
_beginSynchronize(SyncBufferBase& sync_buffer)
{
  auto sync_list = this->m_sync_info->infos();
  Integer nb_message = sync_list.size();

  constexpr int serialize_tag = 523;

  MpiParallelMng* pm= m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();
  const MPI_Datatype mpi_dt = dtlist->datatype(Byte())->datatype();


  double sync_copy_send_time = 0.0;
  double sync_copy_recv_time = 0.0;
  double sync_wait_time = 0.0;

  {
    MpiTimeInterval tit(&sync_copy_send_time);
    for( Integer i=0; i<nb_message; ++i )
      sync_buffer.copySend(i);
  }

  {
    MpiTimeInterval tit(&sync_wait_time);
    for( Integer i=0; i<nb_message; ++i ){
      const VariableSyncInfo& vsi = sync_list[i];
      auto rbuf = sync_buffer.ghostMemoryView(i).bytes().smallView();
      auto sbuf = sync_buffer.shareMemoryView(i).bytes().smallView();

      MPI_Sendrecv(sbuf.data(),
          sbuf.size(),
          mpi_dt,
          vsi.targetRank(),
          serialize_tag,
          rbuf.data(),
          rbuf.size(),
          mpi_dt,
          vsi.targetRank(),
          serialize_tag,
          pm->communicator(),
          MPI_STATUS_IGNORE);

    }
  }

  {
    MpiTimeInterval tit(&sync_copy_recv_time);
    for( Integer i=0; i<nb_message; ++i )
      sync_buffer.copyReceive(i);
  }

  pm->stat()->add("SyncCopySend",sync_copy_send_time,1);
  pm->stat()->add("SyncWait",sync_wait_time,1);
  pm->stat()->add("SyncCopyRecv",sync_copy_recv_time,1);

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiDirectSendrecvVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBufferBase& sync_buffer)
{
  //With this implementation, we do not need this function.
  ARCANE_UNUSED(sync_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiDirectSendrecvVariableSynchronizeDispatcher<Byte>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Int16>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Int32>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Int64>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Real>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Real2>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Real3>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Real2x2>;
template class MpiDirectSendrecvVariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
