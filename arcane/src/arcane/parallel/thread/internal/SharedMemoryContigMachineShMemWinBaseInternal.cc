// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryContigMachineShMemWinBaseInternal.cc            (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/internal/SharedMemoryContigMachineShMemWinBaseInternal.h"

#include "arcane/utils/FatalErrorException.h"

#include "arccore/concurrency/IThreadBarrier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryContigMachineShMemWinBaseInternal::
SharedMemoryContigMachineShMemWinBaseInternal(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<UniqueArray<std::byte>> window, Ref<UniqueArray<Int64>> sizeof_segments, Ref<UniqueArray<Int64>> sum_sizeof_segments, Int64 sizeof_window, IThreadBarrier* barrier)
: m_my_rank(my_rank)
, m_nb_rank(nb_rank)
, m_sizeof_type(sizeof_type)
, m_actual_sizeof_win(sizeof_window)
, m_max_sizeof_win(sizeof_window)
, m_ranks(ranks)
, m_window_span(window->span())
, m_window(window)
, m_sizeof_segments(sizeof_segments)
, m_sizeof_segments_span(sizeof_segments->smallSpan())
, m_sum_sizeof_segments(sum_sizeof_segments)
, m_sum_sizeof_segments_span(sum_sizeof_segments->smallSpan())
, m_barrier(barrier)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 SharedMemoryContigMachineShMemWinBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> SharedMemoryContigMachineShMemWinBaseInternal::
segmentView()
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[m_my_rank];
  const Int64 size_segment = m_sizeof_segments_span[m_my_rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> SharedMemoryContigMachineShMemWinBaseInternal::
segmentView(Int32 rank)
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[rank];
  const Int64 size_segment = m_sizeof_segments_span[rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> SharedMemoryContigMachineShMemWinBaseInternal::
windowView()
{
  return m_window_span;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> SharedMemoryContigMachineShMemWinBaseInternal::
segmentConstView() const
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[m_my_rank];
  const Int64 size_segment = m_sizeof_segments_span[m_my_rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> SharedMemoryContigMachineShMemWinBaseInternal::
segmentConstView(Int32 rank) const
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[rank];
  const Int64 size_segment = m_sizeof_segments_span[rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> SharedMemoryContigMachineShMemWinBaseInternal::
windowConstView() const
{
  return m_window_span;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryContigMachineShMemWinBaseInternal::
resizeSegment(Int64 new_sizeof_segment)
{
  m_sizeof_segments_span[m_my_rank] = new_sizeof_segment;

  m_barrier->wait();

  if (m_my_rank == 0) {
    Int64 sum = 0;
    for (Int32 i = 0; i < m_nb_rank; ++i) {
      m_sum_sizeof_segments_span[i] = sum;
      sum += m_sizeof_segments_span[i];
    }
    if (sum > m_max_sizeof_win) {
      ARCANE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_sizeof_win = sum;
  }
  else {
    Int64 sum = 0;
    for (Int32 i = 0; i < m_nb_rank; ++i) {
      sum += m_sizeof_segments_span[i];
    }
    if (sum > m_max_sizeof_win) {
      ARCANE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_sizeof_win = sum;
  }

  m_window_span = Span<std::byte>{ m_window_span.data(), m_actual_sizeof_win };

  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> SharedMemoryContigMachineShMemWinBaseInternal::
machineRanks() const
{
  return m_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryContigMachineShMemWinBaseInternal::
barrier() const
{
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
