// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiContigMachineShMemWinBaseInternal.cc                     (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of a memory window for a calculation node     */
/* with MPI. This window will be contiguous for all processes on the same    */
/* node.                                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiContigMachineShMemWinBaseInternal.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiContigMachineShMemWinBaseInternal::
MpiContigMachineShMemWinBaseInternal(Int64 sizeof_segment, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win()
, m_win_sizeof_segments()
, m_win_sum_sizeof_segments()
, m_comm_machine(comm_machine)
, m_comm_machine_size(comm_machine_size)
, m_comm_machine_rank(comm_machine_rank)
, m_sizeof_type(sizeof_type)
, m_machine_ranks(machine_ranks)
, m_actual_sizeof_win(-1)
{
  // All windows of this class must be contiguous.
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

  // Allocate the main window (which will contain the user data.
  // We do not retrieve the segment pointer.
  {
    void* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof_segment, m_sizeof_type, win_info, m_comm_machine, &ptr_seg, &m_win);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
  }

  //--------------------------

  // Allocate the window that will contain the size of each segment of the
  // main window.
  {
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof(Int64), sizeof(Int64), win_info, m_comm_machine, &ptr_seg, &m_win_sizeof_segments);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
    // We use the pointer to our segment to set the size of
    // our segment in the main window.
    *ptr_seg = sizeof_segment;
  }

  // Allocate the window that will contain the position of each segment of the
  // main window.
  {
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof(Int64), sizeof(Int64), win_info, m_comm_machine, &ptr_seg, &m_win_sum_sizeof_segments);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
  }

  MPI_Info_free(&win_info);

  MPI_Barrier(m_comm_machine);

  //--------------------------

#ifdef ARCCORE_DEBUG

  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    // Create a view on the entire window containing the sizes.
    // (The loop is only here in debug mode to verify that we have
    // contiguous windows).
    {
      MPI_Aint size_seg;
      int size_type;
      Int64* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_win_sizeof_segments, i, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      if (i == 0) {
        m_sizeof_segments_span = SmallSpan<Int64>{ ptr_seg, m_comm_machine_size };
      }

      if (m_sizeof_segments_span.data() + i != ptr_seg) {
        ARCCORE_FATAL("Segment address error");
      }
      if (m_sizeof_segments_span[i] != *ptr_seg) {
        ARCCORE_FATAL("Segment size error");
      }
    }

    // Create a view on the entire window containing the segment positions.
    {
      MPI_Aint size_seg;
      int size_type;
      Int64* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_win_sum_sizeof_segments, i, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      if (i == 0) {
        m_sum_sizeof_segments_span = SmallSpan<Int64>{ ptr_seg, m_comm_machine_size };
      }

      if (m_sum_sizeof_segments_span.data() + i != ptr_seg) {
        ARCCORE_FATAL("Segment address error");
      }
    }
  }
#else
  // Create a view on the entire window containing the sizes.
  {
    MPI_Aint size_seg;
    int size_type;
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win_sizeof_segments, 0, &size_seg, &size_type, &ptr_seg);
    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_sizeof_segments_span = SmallSpan<Int64>{ ptr_seg, m_comm_machine_size };
  }

  // Create a view on the entire window containing the segment positions.
  {
    MPI_Aint size_seg;
    int size_type;
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win_sum_sizeof_segments, 0, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_sum_sizeof_segments_span = SmallSpan<Int64>{ ptr_seg, m_comm_machine_size };
  }
#endif

  //--------------------------

  // Only process 0 must fill the positions.
  // Everyone calculates the size of the main window.
  if (m_comm_machine_rank == 0) {
    for (Int32 i = 0; i < m_comm_machine_size; ++i) {
      m_sum_sizeof_segments_span[i] = m_max_sizeof_win;
      m_max_sizeof_win += m_sizeof_segments_span[i];
    }
  }
  else {
    for (Int32 i = 0; i < m_comm_machine_size; ++i) {
      m_max_sizeof_win += m_sizeof_segments_span[i];
    }
  }

  MPI_Barrier(m_comm_machine);

  // The current size of the window is its max size.
  // Useful in case of resize.
  m_actual_sizeof_win = m_max_sizeof_win;

  //--------------------------

#ifdef ARCCORE_DEBUG
  Int64 sum = 0;

  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    // Create the view to the main window.
    // (The loop is only here in debug mode to verify that we have
    // a contiguous window).
    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win, i, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }
    if (i == 0) {
      m_window_span = Span<std::byte>{ ptr_seg, m_max_sizeof_win };
    }

    if (ptr_seg != (m_window_span.data() + sum)) {
      ARCCORE_FATAL("Segment address error");
    }
    if (size_seg != m_sizeof_segments_span[i]) {
      ARCCORE_FATAL("Segment size error");
    }
    sum += size_seg;
  }
  if (sum != m_max_sizeof_win) {
    ARCCORE_FATAL("Window size error -- Expected : {0} -- Found : {1}", m_max_sizeof_win, sum);
  }
#else
  // Create the view to the main window.
  {
    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win, 0, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_window_span = Span<std::byte>{ ptr_seg, m_max_sizeof_win };
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiContigMachineShMemWinBaseInternal::
~MpiContigMachineShMemWinBaseInternal()
{
  MPI_Win_free(&m_win);
  MPI_Win_free(&m_win_sizeof_segments);
  MPI_Win_free(&m_win_sum_sizeof_segments);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiContigMachineShMemWinBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiContigMachineShMemWinBaseInternal::
segmentView()
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[m_comm_machine_rank];
  const Int64 size_segment = m_sizeof_segments_span[m_comm_machine_rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiContigMachineShMemWinBaseInternal::
segmentView(Int32 rank)
{
  Int32 pos = -1;
  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == rank) {
      pos = i;
      break;
    }
  }
  if (pos == -1) {
    ARCCORE_FATAL("Rank is not in machine");
  }

  const Int64 begin_segment = m_sum_sizeof_segments_span[pos];
  const Int64 size_segment = m_sizeof_segments_span[pos];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiContigMachineShMemWinBaseInternal::
windowView()
{
  return m_window_span;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiContigMachineShMemWinBaseInternal::
segmentConstView() const
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[m_comm_machine_rank];
  const Int64 size_segment = m_sizeof_segments_span[m_comm_machine_rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiContigMachineShMemWinBaseInternal::
segmentConstView(Int32 rank) const
{
  Int32 pos = -1;
  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == rank) {
      pos = i;
      break;
    }
  }
  if (pos == -1) {
    ARCCORE_FATAL("Rank is not in machine");
  }

  const Int64 begin_segment = m_sum_sizeof_segments_span[pos];
  const Int64 size_segment = m_sizeof_segments_span[pos];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiContigMachineShMemWinBaseInternal::
windowConstView() const
{
  return m_window_span;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiContigMachineShMemWinBaseInternal::
resizeSegment(Int64 new_sizeof_segment)
{
  m_sizeof_segments_span[m_comm_machine_rank] = new_sizeof_segment;

  MPI_Barrier(m_comm_machine);

  if (m_comm_machine_rank == 0) {
    Int64 sum = 0;
    for (Int32 i = 0; i < m_comm_machine_size; ++i) {
      m_sum_sizeof_segments_span[i] = sum;
      sum += m_sizeof_segments_span[i];
    }
    if (sum > m_max_sizeof_win) {
      ARCCORE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_sizeof_win = sum;
  }
  else {
    Int64 sum = 0;
    for (Int32 i = 0; i < m_comm_machine_size; ++i) {
      sum += m_sizeof_segments_span[i];
    }
    if (sum > m_max_sizeof_win) {
      ARCCORE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_sizeof_win = sum;
  }

  m_window_span = Span<std::byte>{ m_window_span.data(), m_actual_sizeof_win };

  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiContigMachineShMemWinBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiContigMachineShMemWinBaseInternal::
barrier() const
{
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
