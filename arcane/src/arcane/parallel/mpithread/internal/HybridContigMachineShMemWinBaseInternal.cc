// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridContigMachineShMemWinBaseInternal.cc                  (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridContigMachineShMemWinBaseInternal.h"

#include "arcane/parallel/mpithread/HybridMessageQueue.h"

#include "arccore/concurrency/IThreadBarrier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridContigMachineShMemWinBaseInternal::
HybridContigMachineShMemWinBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<IContigMachineShMemWinBaseInternal> nb_elem, Ref<IContigMachineShMemWinBaseInternal> sum_nb_elem, Ref<IContigMachineShMemWinBaseInternal> mpi_window, IThreadBarrier* barrier)
: m_my_rank_local_proc(my_rank_local_proc)
, m_nb_rank_local_proc(nb_rank_local_proc)
, m_my_rank_mpi(my_rank_mpi)
, m_machine_ranks(ranks)
, m_sizeof_type(sizeof_type)
, m_mpi_window(mpi_window)
, m_sizeof_sub_segments_global(nb_elem)
, m_sum_sizeof_sub_segments_global(sum_nb_elem)
, m_thread_barrier(barrier)
{
  m_sizeof_sub_segments_local_proc = asSpan<Int64>(m_sizeof_sub_segments_global->segmentView());
  m_sum_sizeof_sub_segments_local_proc = asSpan<Int64>(m_sum_sizeof_sub_segments_global->segmentView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HybridContigMachineShMemWinBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridContigMachineShMemWinBaseInternal::
segmentView()
{
  const Span<std::byte> segment_proc = m_mpi_window->segmentView();
  const Int64 begin_segment_thread = m_sum_sizeof_sub_segments_local_proc[m_my_rank_local_proc];
  const Int64 size_segment_thread = m_sizeof_sub_segments_local_proc[m_my_rank_local_proc];

  return segment_proc.subSpan(begin_segment_thread, size_segment_thread);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridContigMachineShMemWinBaseInternal::
segmentView(Int32 rank)
{
  const FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);

  // Si le rang est un thread de notre processus.
  if (fri.mpiRankValue() == m_my_rank_mpi) {
    const Span<std::byte> segment_proc = m_mpi_window->segmentView();
    const Int64 begin_segment_thread = m_sum_sizeof_sub_segments_local_proc[fri.localRankValue()];
    const Int64 size_segment_thread = m_sizeof_sub_segments_local_proc[fri.localRankValue()];

    return segment_proc.subSpan(begin_segment_thread, size_segment_thread);
  }

  const Span<Int64> sum_nb_elem_other_proc = asSpan<Int64>(m_sum_sizeof_sub_segments_global->segmentView(fri.mpiRankValue()));
  const Span<Int64> nb_elem_other_proc = asSpan<Int64>(m_sizeof_sub_segments_global->segmentView(fri.mpiRankValue()));

  const Span<std::byte> segment_proc = m_mpi_window->segmentView(fri.mpiRankValue());
  const Int64 begin_segment_thread = sum_nb_elem_other_proc[fri.localRankValue()];
  const Int64 size_segment_thread = nb_elem_other_proc[fri.localRankValue()];

  return segment_proc.subSpan(begin_segment_thread, size_segment_thread);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridContigMachineShMemWinBaseInternal::
windowView()
{
  return m_mpi_window->windowView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> HybridContigMachineShMemWinBaseInternal::
segmentConstView() const
{
  const Span<const std::byte> segment_proc = m_mpi_window->segmentConstView();
  const Int64 begin_segment_thread = m_sum_sizeof_sub_segments_local_proc[m_my_rank_local_proc];
  const Int64 size_segment_thread = m_sizeof_sub_segments_local_proc[m_my_rank_local_proc];

  return segment_proc.subSpan(begin_segment_thread, size_segment_thread);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> HybridContigMachineShMemWinBaseInternal::
segmentConstView(Int32 rank) const
{
  const FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);

  // Si le rang est un thread de notre processus.
  if (fri.mpiRankValue() == m_my_rank_mpi) {
    const Span<const std::byte> segment_proc = m_mpi_window->segmentConstView();
    const Int64 begin_segment_thread = m_sum_sizeof_sub_segments_local_proc[fri.localRankValue()];
    const Int64 size_segment_thread = m_sizeof_sub_segments_local_proc[fri.localRankValue()];

    return segment_proc.subSpan(begin_segment_thread, size_segment_thread);
  }

  const Span<const Int64> sum_nb_elem_other_proc = asSpan<const Int64>(m_sum_sizeof_sub_segments_global->segmentConstView(fri.mpiRankValue()));
  const Span<const Int64> nb_elem_other_proc = asSpan<const Int64>(m_sizeof_sub_segments_global->segmentConstView(fri.mpiRankValue()));

  const Span<const std::byte> segment_proc = m_mpi_window->segmentConstView(fri.mpiRankValue());
  const Int64 begin_segment_thread = sum_nb_elem_other_proc[fri.localRankValue()];
  const Int64 size_segment_thread = nb_elem_other_proc[fri.localRankValue()];

  return segment_proc.subSpan(begin_segment_thread, size_segment_thread);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> HybridContigMachineShMemWinBaseInternal::
windowConstView() const
{
  return m_mpi_window->windowConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridContigMachineShMemWinBaseInternal::
resizeSegment(Int64 new_sizeof_segment)
{
  m_sizeof_sub_segments_local_proc[m_my_rank_local_proc] = new_sizeof_segment;

  m_thread_barrier->wait();

  if (m_my_rank_local_proc == 0) {
    Int64 sum = 0;
    for (Int32 i = 0; i < m_nb_rank_local_proc; ++i) {
      m_sum_sizeof_sub_segments_local_proc[i] = sum;
      sum += m_sizeof_sub_segments_local_proc[i];
    }
    m_mpi_window->resizeSegment(sum);
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> HybridContigMachineShMemWinBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridContigMachineShMemWinBaseInternal::
barrier() const
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0)
    m_mpi_window->barrier();
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
