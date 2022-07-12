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

template<typename SimpleType>
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
MpiNeighborVariableSynchronizeDispatcher(MpiNeighborVariableSynchronizeDispatcherBuildInfo& bi)
: VariableSynchronizeDispatcher<SimpleType>(VariableSynchronizeDispatcherBuildInfo(bi.parallelMng(),bi.table()))
, m_mpi_parallel_mng(bi.parallelMng())
, m_neighbor_communicator(MPI_COMM_NULL)
, m_synchronizer_communicator(bi.synchronizerCommunicator())
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
  // Ne fait rien au niveau MPI dans cette partie car cette implémentation
  // ne supporte pas encore l'asynchronisme.
  // On se contente de recopier les valeurs des variables dans le buffer d'envoi
  // pour permettre ensuite de modifier les valeurs de la variable entre
  // le _beginSynchronize() et le _endSynchronize().

  double send_copy_time = 0.0;
  {
    MpiTimeInterval tit(&send_copy_time);

    // Recopie les buffers d'envoi
    auto sync_list = this->m_sync_info->infos();
    Integer nb_message = sync_list.size();
    for (Integer i = 0; i < nb_message; ++i)
      sync_buffer.copySend(i);
  }
  Int64 total_share_size = sync_buffer.totalShareSize();
  m_mpi_parallel_mng->stat()->add("SyncSendCopy", send_copy_time, total_share_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBuffer& sync_buffer)
{
  auto sync_list = this->m_sync_info->infos();
  const Int32 nb_message = sync_list.size();

  auto* sync_communicator = m_synchronizer_communicator.get();
  ARCANE_CHECK_POINTER(sync_communicator);

  MPI_Comm communicator = sync_communicator->communicator();
  if (communicator == MPI_COMM_NULL)
    ARCANE_FATAL("Invalid null communicator");

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();

  double copy_time = 0.0;
  double wait_time = 0.0;

  const MPI_Datatype mpi_dt = dtlist->datatype(SimpleType())->datatype();

  ConstArrayView<SimpleType> send_buf = sync_buffer.shareBuffer();
  ArrayView<SimpleType> receive_buf = sync_buffer.ghostBuffer();

  for (Integer i = 0; i < nb_message; ++i) {
    // TODO: vérifier débordement
    Int32 nb_send = sync_buffer.shareBuffer(i).size();
    Int32 nb_receive = sync_buffer.ghostBuffer(i).size();
    m_mpi_send_counts[i] = nb_send;
    m_mpi_receive_counts[i] = nb_receive;
    m_mpi_send_displacements[i] = sync_buffer.shareDisplacement(i);
    m_mpi_receive_displacements[i] = sync_buffer.ghostDisplacement(i);
  }

  {
    MpiTimeInterval tit(&wait_time);
    MPI_Neighbor_alltoallv(send_buf.data(), m_mpi_send_counts.data(), m_mpi_send_displacements.data(), mpi_dt,
                           receive_buf.data(), m_mpi_receive_counts.data(), m_mpi_receive_displacements.data(), mpi_dt,
                           communicator);
  }

  // Recopie les valeurs recues
  {
    MpiTimeInterval tit(&copy_time);
    for (Integer i = 0; i < nb_message; ++i)
      sync_buffer.copyReceive(i);
  }

  Int64 total_ghost_size = sync_buffer.totalGhostSize();
  Int64 total_share_size = sync_buffer.totalShareSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy", copy_time, total_ghost_size);
  pm->stat()->add("SyncWait", wait_time, total_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiNeighborVariableSynchronizeDispatcher<SimpleType>::
compute()
{
  VariableSynchronizeDispatcher<SimpleType>::compute();

  auto* sync_communicator = m_synchronizer_communicator.get();
  ARCANE_CHECK_POINTER(sync_communicator);

  auto sync_list = this->m_sync_info->infos();
  const Int32 nb_message = sync_list.size();

  m_mpi_send_counts.resize(nb_message);
  m_mpi_receive_counts.resize(nb_message);
  m_mpi_send_displacements.resize(nb_message);
  m_mpi_receive_displacements.resize(nb_message);
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
