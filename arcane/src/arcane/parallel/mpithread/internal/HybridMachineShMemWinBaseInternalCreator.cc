// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineShMemWinBaseInternalCreator.cc                 (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of objects of type                            */
/* HybridContigMachineShMemWinBaseInternal. An instance of this object must  */
/* be shared by all threads of a process.                                    */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridMachineShMemWinBaseInternalCreator.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpithread/internal/HybridContigMachineShMemWinBaseInternal.h"
#include "arcane/parallel/mpithread/internal/HybridMachineShMemWinBaseInternal.h"
#include "arcane/parallel/mpithread/HybridMessageQueue.h"

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing_mpi/internal/MpiMachineShMemWinBaseInternalCreator.h"
#include "arccore/message_passing_mpi/internal/MpiContigMachineShMemWinBaseInternal.h"
#include "arccore/message_passing_mpi/internal/MpiMultiMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineShMemWinBaseInternalCreator::
HybridMachineShMemWinBaseInternalCreator(Int32 nb_rank_local_proc, IThreadBarrier* barrier)
: m_nb_rank_local_proc(nb_rank_local_proc)
, m_barrier(barrier)
, m_sizeof_resize_segments(nb_rank_local_proc)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternalCreator::
initializeMpiWindowCreator(Int32 my_rank_global, MpiParallelMng* mpi_parallel_mng)
{
  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(my_rank_global), m_nb_rank_local_proc);
  Int32 my_rank_local_proc = my_fri.localRankValue();
  if (my_rank_local_proc == 0) {
    // mpi_parallel_mng->adapter()->initializeWindowCreator(mpi_parallel_mng->machineCommunicator()); // Managed by MpiParallelMng
    _buildMachineRanksArray(mpi_parallel_mng->adapter()->windowCreator()->machineRanks());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridContigMachineShMemWinBaseInternal* HybridMachineShMemWinBaseInternalCreator::
createWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng)
{
  // We are in a context where each process must have several segments, one per thread.
  // For each process to have access to all segment positions from all threads of all processes,
  // each process must share the positions of these segments with the other processes.
  // To do this, we use MPI memory windows.

  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(my_rank_global), m_nb_rank_local_proc);
  Int32 my_rank_local_proc = my_fri.localRankValue();
  Int32 my_rank_mpi = my_fri.mpiRankValue();

  Mpi::MpiMachineShMemWinBaseInternalCreator* mpi_window_creator = nullptr;

  if (my_rank_local_proc == 0) {
    mpi_window_creator = mpi_parallel_mng->adapter()->windowCreator();

    // The number of elements in each segment. This window will be sized
    // nb_thread * nb_proc_on_the_same_node.
    m_sizeof_sub_segments = makeRef(mpi_window_creator->createWindow(m_nb_rank_local_proc * static_cast<Int64>(sizeof(Int64)), sizeof(Int64)));
    m_sum_sizeof_sub_segments = makeRef(mpi_window_creator->createWindow(m_nb_rank_local_proc * static_cast<Int64>(sizeof(Int64)), sizeof(Int64)));
  }
  m_barrier->wait();

  // nb_elem is the segment of our process (which contains the
  // segments of all our threads).
  Span<Int64> sizeof_sub_segments = asSpan<Int64>(m_sizeof_sub_segments->segmentView());

  sizeof_sub_segments[my_rank_local_proc] = sizeof_segment;
  m_barrier->wait();

  if (my_rank_local_proc == 0) {
    m_sizeof_segment_local_proc = 0;
    Span<Int64> sum_sizeof_sub_segments = asSpan<Int64>(m_sum_sizeof_sub_segments->segmentView());

    for (Int32 i = 0; i < m_nb_rank_local_proc; ++i) {
      sum_sizeof_sub_segments[i] = m_sizeof_segment_local_proc;
      m_sizeof_segment_local_proc += sizeof_sub_segments[i];
    }
  }
  m_barrier->wait();

  if (my_rank_local_proc == 0) {
    m_window = makeRef(mpi_window_creator->createWindow(m_sizeof_segment_local_proc, sizeof_type));
  }
  m_barrier->wait();

  auto* window_obj = new HybridContigMachineShMemWinBaseInternal(my_rank_mpi, my_rank_local_proc, m_nb_rank_local_proc, m_machine_ranks, sizeof_type, m_sizeof_sub_segments, m_sum_sizeof_sub_segments, m_window, m_barrier);
  m_barrier->wait();

  // These arrays must be deleted by HybridContigMachineShMemWinBaseInternal.
  if (my_rank_local_proc == 0) {
    m_sizeof_sub_segments.reset();
    m_sum_sizeof_sub_segments.reset();
    m_sizeof_segment_local_proc = 0;
    m_window.reset();
  }

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineShMemWinBaseInternal* HybridMachineShMemWinBaseInternalCreator::
createDynamicWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng)
{
  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(my_rank_global), m_nb_rank_local_proc);
  Int32 my_rank_local_proc = my_fri.localRankValue();
  Int32 my_rank_mpi = my_fri.mpiRankValue();

  Mpi::MpiMachineShMemWinBaseInternalCreator* mpi_window_creator = nullptr;

  if (my_rank_local_proc == 0) {
    mpi_window_creator = mpi_parallel_mng->adapter()->windowCreator();
  }
  m_barrier->wait();

  m_sizeof_resize_segments[my_rank_local_proc] = sizeof_segment;
  m_barrier->wait();

  if (my_rank_local_proc == 0) {
    m_windows = makeRef(mpi_window_creator->createDynamicMultiWindow(m_sizeof_resize_segments.smallSpan(), m_nb_rank_local_proc, sizeof_type));
    m_sizeof_resize_segments.fill(0);
  }
  m_barrier->wait();

  auto* window_obj = new HybridMachineShMemWinBaseInternal(my_rank_mpi, my_rank_local_proc, m_nb_rank_local_proc, m_machine_ranks, sizeof_type, m_windows, m_barrier);
  m_barrier->wait();

  // These arrays must be deleted by HybridMachineShMemWinBaseInternal.
  if (my_rank_local_proc == 0) {
    m_windows.reset();
  }

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> HybridMachineShMemWinBaseInternalCreator::
machineRanks()
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternalCreator::
machineBarrier(Int32 my_rank_global, MpiParallelMng* mpi_parallel_mng) const
{
  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(my_rank_global), m_nb_rank_local_proc);
  Int32 my_rank_local_proc = my_fri.localRankValue();
  m_barrier->wait();
  if (my_rank_local_proc == 0)
    mpi_parallel_mng->barrier();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternalCreator::
_buildMachineRanksArray(ConstArrayView<Int32> mpi_machine_ranks)
{
  m_machine_ranks.resize(mpi_machine_ranks.size() * m_nb_rank_local_proc);

  Int32 iter = 0;
  for (Int32 mpi_rank : mpi_machine_ranks) {
    for (Int32 thread_rank = 0; thread_rank < m_nb_rank_local_proc; ++thread_rank) {
      m_machine_ranks[iter++] = thread_rank + m_nb_rank_local_proc * mpi_rank;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
