// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBaseInternalCreator.cc             (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* HybridMachineMemoryWindowBaseInternal. Une instance de cet objet doit     */
/* être partagée par tous les threads d'un processus.                        */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBaseInternalCreator.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBaseInternal.h"
#include "arcane/parallel/mpithread/HybridMessageQueue.h"

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternalCreator.h"
#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineMemoryWindowBaseInternalCreator::
HybridMachineMemoryWindowBaseInternalCreator(Int32 nb_rank_local_proc, IThreadBarrier* barrier)
: m_nb_rank_local_proc(nb_rank_local_proc)
, m_sizeof_segment_local_proc(0)
, m_barrier(barrier)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineMemoryWindowBaseInternal* HybridMachineMemoryWindowBaseInternalCreator::
createWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng)
{
  // On est dans un contexte où chaque processus doit avoir plusieurs segments, un par thread.
  // Pour que chaque processus puisse avoir accès à toutes les positions des segments de tous les
  // threads de tous les processus, chaque processus doit partager les positions de ces segments avec
  // les autres processus. Pour faire ça, on utilise des fenêtres mémoire MPI.

  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(my_rank_global), m_nb_rank_local_proc);
  Int32 my_rank_local_proc = my_fri.localRankValue();
  Int32 my_rank_mpi = my_fri.mpiRankValue();

  Mpi::MpiMachineMemoryWindowBaseInternalCreator* mpi_window_creator = nullptr;

  if (my_rank_local_proc == 0) {
    mpi_window_creator = mpi_parallel_mng->adapter()->windowCreator();
    _buildMachineRanksArray(mpi_window_creator);

    // Le nombre d'éléments de chaque segment. Cette fenêtre fera une taille de nb_thread * nb_proc_sur_le_même_noeud.
    m_sizeof_sub_segments = makeRef(mpi_window_creator->createWindow(m_nb_rank_local_proc * static_cast<Int64>(sizeof(Int64)), sizeof(Int64)));
    m_sum_sizeof_sub_segments = makeRef(mpi_window_creator->createWindow(m_nb_rank_local_proc * static_cast<Int64>(sizeof(Int64)), sizeof(Int64)));
  }
  m_barrier->wait();

  // nb_elem est le segment de notre processus (qui contient les segments de tous nos threads).
  Span<Int64> sizeof_sub_segments = asSpan<Int64>(m_sizeof_sub_segments->segment());

  sizeof_sub_segments[my_rank_local_proc] = sizeof_segment;
  m_barrier->wait();

  if (my_rank_local_proc == 0) {
    m_sizeof_segment_local_proc = 0;
    Span<Int64> sum_sizeof_sub_segments = asSpan<Int64>(m_sum_sizeof_sub_segments->segment());

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

  auto* window_obj = new HybridMachineMemoryWindowBaseInternal(my_rank_mpi, my_rank_local_proc, m_nb_rank_local_proc, m_machine_ranks, sizeof_type, m_sizeof_sub_segments, m_sum_sizeof_sub_segments, m_window, m_barrier);
  m_barrier->wait();

  // Ces tableaux doivent être delete par HybridMachineMemoryWindowBaseInternal (rang 0 uniquement).
  m_sizeof_sub_segments.reset();
  m_sum_sizeof_sub_segments.reset();
  m_sizeof_segment_local_proc = 0;
  m_window.reset();

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineMemoryWindowBaseInternalCreator::
_buildMachineRanksArray(const Mpi::MpiMachineMemoryWindowBaseInternalCreator* mpi_window_creator)
{
  ConstArrayView<Int32> mpi_ranks(mpi_window_creator->machineRanks());
  m_machine_ranks.resize(mpi_ranks.size() * m_nb_rank_local_proc);

  Int32 iter = 0;
  for (Int32 mpi_rank : mpi_ranks) {
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
