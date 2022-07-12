// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiNeighborVariableSynchronizeDispatcher.cc                 (C) 2000-2022 */
/*                                                                           */
/* Synchronisations des variables via MPI_Neighbor_alltoallv.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/MpiNeighborVariableSynchronizeDispatcher.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiAdapter.h"
#include "arcane/parallel/mpi/MpiDatatypeList.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/MpiTimeInterval.h"
#include "arcane/parallel/IStat.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Cette implémentation utilise la fonction MPI_Neighbor_alltoallv pour
 * les synchronisations. Cette fonction est disponible dans la version 3.1
 * de MPI.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GenericMpiNeighborVariableSynchronizer::
GenericMpiNeighborVariableSynchronizer(MpiNeighborVariableSynchronizeDispatcherBuildInfo& bi)
: m_mpi_parallel_mng(bi.parallelMng())
, m_synchronizer_communicator(bi.synchronizerCommunicator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericMpiNeighborVariableSynchronizer::
beginSynchronize(IVariableSynchronizerBuffer* buf)
{
  // Ne fait rien au niveau MPI dans cette partie car cette implémentation
  // ne supporte pas encore l'asynchronisme.
  // On se contente de recopier les valeurs des variables dans le buffer d'envoi
  // pour permettre ensuite de modifier les valeurs de la variable entre
  // le beginSynchronize() et le endSynchronize().

  double send_copy_time = 0.0;
  {
    MpiTimeInterval tit(&send_copy_time);

    // Recopie les buffers d'envoi
    Integer nb_message = buf->nbRank();
    for (Integer i = 0; i < nb_message; ++i)
      buf->copySend(i);
  }
  Int64 total_share_size = buf->totalSendSize();
  m_mpi_parallel_mng->stat()->add("SyncSendCopy", send_copy_time, total_share_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericMpiNeighborVariableSynchronizer::
endSynchronize(IVariableSynchronizerBuffer* buf)
{
  const Int32 nb_message = buf->nbRank();

  auto* sync_communicator = m_synchronizer_communicator.get();
  ARCANE_CHECK_POINTER(sync_communicator);

  MPI_Comm communicator = sync_communicator->communicator();
  if (communicator == MPI_COMM_NULL)
    ARCANE_FATAL("Invalid null communicator");

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();

  double copy_time = 0.0;
  double wait_time = 0.0;

  const MPI_Datatype mpi_dt = dtlist->datatype(Byte())->datatype();

  for (Integer i = 0; i < nb_message; ++i) {
    Int32 nb_send = CheckedConvert::toInt32(buf->sendBuffer(i).size());
    Int32 nb_receive = CheckedConvert::toInt32(buf->receiveBuffer(i).size());
    Int32 send_displacement = CheckedConvert::toInt32(buf->sendDisplacement(i));
    Int32 receive_displacement = CheckedConvert::toInt32(buf->receiveDisplacement(i));

    m_mpi_send_counts[i] = nb_send;
    m_mpi_receive_counts[i] = nb_receive;
    m_mpi_send_displacements[i] = send_displacement;
    m_mpi_receive_displacements[i] = receive_displacement;
  }

  {
    MpiTimeInterval tit(&wait_time);
    auto send_buf = buf->globalSendBuffer();
    auto receive_buf = buf->globalReceiveBuffer();
    MPI_Neighbor_alltoallv(send_buf.data(), m_mpi_send_counts.data(), m_mpi_send_displacements.data(), mpi_dt,
                           receive_buf.data(), m_mpi_receive_counts.data(), m_mpi_receive_displacements.data(), mpi_dt,
                           communicator);
  }

  // Recopie les valeurs recues
  {
    MpiTimeInterval tit(&copy_time);
    for (Integer i = 0; i < nb_message; ++i)
      buf->copyReceive(i);
  }

  Int64 total_ghost_size = buf->totalReceiveSize();
  Int64 total_share_size = buf->totalSendSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy", copy_time, total_ghost_size);
  pm->stat()->add("SyncWait", wait_time, total_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericMpiNeighborVariableSynchronizer::
compute()
{
  ARCANE_CHECK_POINTER(m_sync_info);

  auto* sync_communicator = m_synchronizer_communicator.get();
  ARCANE_CHECK_POINTER(sync_communicator);

  auto sync_list = m_sync_info->infos();
  const Int32 nb_message = sync_list.size();

  m_mpi_send_counts.resize(nb_message);
  m_mpi_receive_counts.resize(nb_message);
  m_mpi_send_displacements.resize(nb_message);
  m_mpi_receive_displacements.resize(nb_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType>
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
MpiNeighborVariableSynchronizeDispatcher(MpiNeighborVariableSynchronizeDispatcherBuildInfo& bi)
: VariableSynchronizeDispatcher<SimpleType>(VariableSynchronizeDispatcherBuildInfo(bi.parallelMng(), bi.table()))
, m_generic(bi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
_beginSynchronize(SyncBuffer& sync_buffer)
{
  m_generic.beginSynchronize(sync_buffer.genericBuffer());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBuffer& sync_buffer)
{
  m_generic.endSynchronize(sync_buffer.genericBuffer());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
compute()
{
  VariableSynchronizeDispatcher<SimpleType>::compute();
  m_generic.compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiNeighborVariableSynchronizeDispatcher<Byte>;
template class MpiNeighborVariableSynchronizeDispatcher<Int16>;
template class MpiNeighborVariableSynchronizeDispatcher<Int32>;
template class MpiNeighborVariableSynchronizeDispatcher<Int64>;
template class MpiNeighborVariableSynchronizeDispatcher<Real>;
template class MpiNeighborVariableSynchronizeDispatcher<Real2>;
template class MpiNeighborVariableSynchronizeDispatcher<Real3>;
template class MpiNeighborVariableSynchronizeDispatcher<Real2x2>;
template class MpiNeighborVariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
