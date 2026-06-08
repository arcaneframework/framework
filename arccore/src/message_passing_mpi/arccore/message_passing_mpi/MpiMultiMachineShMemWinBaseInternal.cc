// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMultiMachineShMemWinBaseInternal.cc                      (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of memory windows for a compute node.         */
/* The segments of these windows are not contiguous in memory and can        */
/* be resized. A process can possess multiple segments.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiMultiMachineShMemWinBaseInternal.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! sizeof_segments should not be preserved!
MpiMultiMachineShMemWinBaseInternal::
MpiMultiMachineShMemWinBaseInternal(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win_need_resize()
, m_win_actual_sizeof()
, m_win_target_segments()
, m_comm_machine(comm_machine)
, m_comm_machine_size(comm_machine_size)
, m_comm_machine_rank(comm_machine_rank)
, m_sizeof_type(sizeof_type)
, m_nb_segments_per_proc(nb_segments_per_proc)
, m_machine_ranks(machine_ranks)
, m_add_requests(nb_segments_per_proc)
, m_resize_requests(nb_segments_per_proc)
{
  if (m_sizeof_type <= 0) {
    ARCCORE_FATAL("Invalid sizeof_type");
  }
  for (Integer i = 0; i < m_nb_segments_per_proc; ++i) {
    if (sizeof_segments[i] < 0 || sizeof_segments[i] % m_sizeof_type != 0) {
      ARCCORE_FATAL("Invalid initial sizeof_segment");
    }
  }
  if (m_nb_segments_per_proc <= 0) {
    ARCCORE_FATAL("Invalid nb_segments_per_proc");
  }
  m_all_mpi_win.resize(m_comm_machine_size * m_nb_segments_per_proc);
  m_reserved_part_span.resize(m_nb_segments_per_proc);

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    m_add_requests[num_seg] = Span<const std::byte>{ nullptr, 0 };
    m_resize_requests[num_seg] = -1;
  }

  MPI_Info win_info_true;
  MPI_Info_create(&win_info_true);
  MPI_Info_set(win_info_true, "alloc_shared_noncontig", "true");

  MPI_Info win_info_false;
  MPI_Info_create(&win_info_false);
  MPI_Info_set(win_info_false, "alloc_shared_noncontig", "false");

  const Int32 pos_my_wins = m_comm_machine_rank * m_nb_segments_per_proc;

  {
    // We create all segments for all processes.
    for (Integer i = 0; i < m_comm_machine_size; ++i) {
      for (Integer j = 0; j < m_nb_segments_per_proc; ++j) {
        Int64 size_seg = 0;
        if (m_comm_machine_rank == i) {
          if (sizeof_segments[j] == 0)
            size_seg = m_sizeof_type;
          else
            size_seg = sizeof_segments[j];
        }
        std::byte* ptr_seg = nullptr;
        int error = MPI_Win_allocate_shared(size_seg, m_sizeof_type, win_info_true, m_comm_machine, &ptr_seg, &m_all_mpi_win[j + i * m_nb_segments_per_proc]);

        if (error != MPI_SUCCESS) {
          ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
        }
      }
    }

    for (Integer i = 0; i < m_nb_segments_per_proc; ++i) {
      MPI_Aint size_seg;
      int size_type;
      std::byte* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_all_mpi_win[i + pos_my_wins], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      // Attention: The user requests a minimum number of reserved elements.
      // But MPI reserves the size it wants (effect of alloc_shared_noncontig=true).
      // We are just sure that the size it reserved is greater than or equal to sizeof_segment.
      m_reserved_part_span[i] = Span<std::byte>{ ptr_seg, size_seg };
    }
  }

  {
    Int64* ptr_seg = nullptr;
    Int64* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(static_cast<Int64>(sizeof(Int64)) * m_nb_segments_per_proc, sizeof(Int64), win_info_false, m_comm_machine, &ptr_seg, &m_win_need_resize);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_need_resize, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      m_need_resize = Span<Int64>{ ptr_win, m_comm_machine_size * m_nb_segments_per_proc };

      for (Integer i = 0; i < m_nb_segments_per_proc; ++i) {
        m_need_resize[i + pos_my_wins] = -1;
      }
    }
    if (ptr_win + pos_my_wins != ptr_seg) {
      ARCCORE_FATAL("m_win_need_resize is noncontig");
    }
  }

  {
    Int64* ptr_seg = nullptr;
    Int64* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(static_cast<Int64>(sizeof(Int64)) * m_nb_segments_per_proc, sizeof(Int64), win_info_false, m_comm_machine, &ptr_seg, &m_win_actual_sizeof);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_actual_sizeof, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      m_sizeof_used_part = Span<Int64>{ ptr_win, m_comm_machine_size * m_nb_segments_per_proc };

      for (Integer i = 0; i < m_nb_segments_per_proc; ++i) {
        m_sizeof_used_part[i + pos_my_wins] = sizeof_segments[i];
      }
    }
    if (ptr_win + pos_my_wins != ptr_seg) {
      ARCCORE_FATAL("m_win_actual_sizeof is noncontig");
    }
  }

  {
    Int32* ptr_seg = nullptr;
    Int32* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(static_cast<Int64>(sizeof(Int32)) * m_nb_segments_per_proc, sizeof(Int32), win_info_false, m_comm_machine, &ptr_seg, &m_win_target_segments);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_target_segments, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      m_target_segments = Span<Int32>{ ptr_win, m_comm_machine_size * m_nb_segments_per_proc };

      for (Integer i = 0; i < m_nb_segments_per_proc; ++i) {
        m_target_segments[i + pos_my_wins] = -1;
      }
    }
    if (ptr_win + pos_my_wins != ptr_seg) {
      ARCCORE_FATAL("m_win_owner_segments is noncontig");
    }
  }

  MPI_Info_free(&win_info_false);
  MPI_Info_free(&win_info_true);

  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMultiMachineShMemWinBaseInternal::
~MpiMultiMachineShMemWinBaseInternal()
{
  for (Integer i = 0; i < m_comm_machine_size * m_nb_segments_per_proc; ++i) {
    MPI_Win_free(&m_all_mpi_win[i]);
  }
  MPI_Win_free(&m_win_need_resize);
  MPI_Win_free(&m_win_actual_sizeof);
  MPI_Win_free(&m_win_target_segments);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiMultiMachineShMemWinBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiMultiMachineShMemWinBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
barrier() const
{
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiMultiMachineShMemWinBaseInternal::
segmentView(Int32 num_seg)
{
  const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  return m_reserved_part_span[num_seg].subSpan(0, m_sizeof_used_part[segment_infos_pos]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiMultiMachineShMemWinBaseInternal::
segmentView(Int32 rank, Int32 num_seg)
{
  const Int32 segment_infos_pos = num_seg + _worldToMachine(rank) * m_nb_segments_per_proc;

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[segment_infos_pos], rank, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
  }

  return Span<std::byte>{ ptr_seg, m_sizeof_used_part[segment_infos_pos] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiMultiMachineShMemWinBaseInternal::
segmentConstView(Int32 num_seg) const
{
  const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  return m_reserved_part_span[num_seg].subSpan(0, m_sizeof_used_part[segment_infos_pos]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiMultiMachineShMemWinBaseInternal::
segmentConstView(Int32 rank, Int32 num_seg) const
{
  const Int32 segment_infos_pos = num_seg + _worldToMachine(rank) * m_nb_segments_per_proc;

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[segment_infos_pos], rank, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
  }

  return Span<const std::byte>{ ptr_seg, m_sizeof_used_part[segment_infos_pos] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
requestAdd(Int32 num_seg, Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }
  if (elem.empty() || elem.data() == nullptr) {
    return;
  }

  const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

  const Int64 actual_sizeof_win = m_sizeof_used_part[segment_infos_pos];
  const Int64 future_sizeof_win = actual_sizeof_win + elem.size();
  const Int64 old_reserved = m_reserved_part_span[num_seg].size();

  if (future_sizeof_win > old_reserved) {
    _requestRealloc(segment_infos_pos, future_sizeof_win);
  }
  else {
    _requestRealloc(segment_infos_pos);
  }

  m_add_requests[num_seg] = elem;
  m_add_requested = true; // TODO Atomic ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
executeAdd()
{
  _executeRealloc();

  if (!m_add_requested) {
    return;
  }
  m_add_requested = false;

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    if (m_add_requests[num_seg].empty() || m_add_requests[num_seg].data() == nullptr) {
      continue;
    }

    const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    const Int64 actual_sizeof_win = m_sizeof_used_part[segment_infos_pos];
    const Int64 future_sizeof_win = actual_sizeof_win + m_add_requests[num_seg].size();

    if (m_reserved_part_span[num_seg].size() < future_sizeof_win) {
      ARCCORE_FATAL("Bad realloc -- New size : {1} -- Needed size : {2}", m_reserved_part_span[num_seg].size(), future_sizeof_win);
    }

    for (Int64 pos_win = actual_sizeof_win, pos_elem = 0; pos_win < future_sizeof_win; ++pos_win, ++pos_elem) {
      m_reserved_part_span[num_seg][pos_win] = m_add_requests[num_seg][pos_elem];
    }
    m_sizeof_used_part[segment_infos_pos] = future_sizeof_win;

    m_add_requests[num_seg] = Span<const std::byte>{ nullptr, 0 };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
requestAddToAnotherSegment(Int32 thread, Int32 rank, Int32 num_seg, Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }
  if (elem.empty() || elem.data() == nullptr) {
    return;
  }

  const Int32 machine_rank = _worldToMachine(rank);
  const Int32 target_segment_infos_pos = num_seg + machine_rank * m_nb_segments_per_proc;

  {
    const Int32 segment_infos_pos = thread + m_comm_machine_rank * m_nb_segments_per_proc;
    m_target_segments[segment_infos_pos] = target_segment_infos_pos;
  }

  Span<std::byte> rank_reserved_part_span;
  {
    MPI_Aint size_seg;
    std::byte* ptr_seg = nullptr;
    int size_type;
    int error = MPI_Win_shared_query(m_all_mpi_win[target_segment_infos_pos], machine_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }
    rank_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
  }

  const Int64 actual_sizeof_win = m_sizeof_used_part[target_segment_infos_pos];
  const Int64 future_sizeof_win = actual_sizeof_win + elem.size();
  const Int64 old_reserved = rank_reserved_part_span.size();

  if (future_sizeof_win > old_reserved) {
    _requestRealloc(target_segment_infos_pos, future_sizeof_win);
  }
  else {
    _requestRealloc(target_segment_infos_pos);
  }

  m_add_requests[thread] = elem;
  m_add_requested = true; // TODO Atomic ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
executeAddToAnotherSegment()
{
  MPI_Barrier(m_comm_machine);

  // For each segment belonging to my process, I check if someone
  // wants to modify it.
  // is_my_seg_edited will be used at the end of the method.
  auto is_my_seg_edited = std::make_unique<bool[]>(m_comm_machine_size);
  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    for (const Int32 rank_asked : m_target_segments) {
      if (rank_asked == m_comm_machine_rank) {
        is_my_seg_edited[num_seg] = true;
        break;
      }
    }
  }

  // If I have not requested a segment modification, I only perform
  // the reallocations if needed (other processes may request reallocations from me).
  if (!m_add_requested) {
    _executeRealloc();
  }

  else {
    m_add_requested = false;

    // A segment can only be modified by one thread at a time. We check this here:
    for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
      const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
      const Int32 seg_needs_to_edit = m_target_segments[segment_infos_pos];
      if (seg_needs_to_edit == -1)
        continue;

      bool is_found = false;
      for (const Int32 rank_asked : m_target_segments) {
        if (rank_asked == seg_needs_to_edit) {
          if (!is_found) {
            is_found = true;
          }
          else {
            ARCCORE_FATAL("Two subdomains ask same rank for addToAnotherSegment()");
          }
        }
      }
    }

    _executeRealloc();

    // We add the elements into the segments.
    for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
      if (m_add_requests[num_seg].empty() || m_add_requests[num_seg].data() == nullptr) {
        continue;
      }

      const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
      const Int32 target_segment_infos_pos = m_target_segments[segment_infos_pos];
      if (target_segment_infos_pos == -1) {
        ARCCORE_FATAL("Should not go here");
      }

      const Int64 actual_sizeof_win = m_sizeof_used_part[target_segment_infos_pos];
      const Int64 future_sizeof_win = actual_sizeof_win + m_add_requests[num_seg].size();

      // We retrieve the segment to modify.
      Span<std::byte> rank_reserved_part_span;
      {
        MPI_Aint size_seg;
        std::byte* ptr_seg = nullptr;
        int size_type;
        int error = MPI_Win_shared_query(m_all_mpi_win[target_segment_infos_pos], target_segment_infos_pos / m_nb_segments_per_proc, &size_seg, &size_type, &ptr_seg);

        if (error != MPI_SUCCESS) {
          ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
        }
        rank_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
      }

      if (rank_reserved_part_span.size() < future_sizeof_win) {
        ARCCORE_FATAL("Bad realloc -- New size : {1} -- Needed size : {2}", rank_reserved_part_span.size(), future_sizeof_win);
      }

      // We modify it with the elements given in the requestAddToAnotherSegment() method.
      for (Int64 pos_win = actual_sizeof_win, pos_elem = 0; pos_win < future_sizeof_win; ++pos_win, ++pos_elem) {
        rank_reserved_part_span[pos_win] = m_add_requests[num_seg][pos_elem];
      }
      m_sizeof_used_part[target_segment_infos_pos] = future_sizeof_win;

      m_add_requests[num_seg] = Span<const std::byte>{ nullptr, 0 };
      m_target_segments[segment_infos_pos] = -1;
    }
  }
  MPI_Barrier(m_comm_machine);

  // Since others may have modified our segments, we recreate the views.
  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    if (is_my_seg_edited[num_seg]) {
      const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

      MPI_Aint size_seg;
      std::byte* ptr_seg = nullptr;
      int size_type;
      int error = MPI_Win_shared_query(m_all_mpi_win[segment_infos_pos], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      m_reserved_part_span[num_seg] = Span<std::byte>{ ptr_seg, size_seg };
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
requestReserve(Int32 num_seg, Int64 new_capacity)
{
  if (new_capacity % m_sizeof_type) {
    ARCCORE_FATAL("new_capacity not valid");
  }

  const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

  if (new_capacity > m_reserved_part_span[num_seg].size()) {
    _requestRealloc(segment_infos_pos, new_capacity);
  }
  else {
    _requestRealloc(segment_infos_pos);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
executeReserve()
{
  _executeRealloc();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
requestResize(Int32 num_seg, Int64 new_size)
{
  if (new_size == -1) {
    return;
  }
  if (new_size < 0 || new_size % m_sizeof_type) {
    ARCCORE_FATAL("new_size not valid");
  }

  const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

  if (new_size > m_reserved_part_span[num_seg].size()) {
    _requestRealloc(segment_infos_pos, new_size);
  }
  else {
    _requestRealloc(segment_infos_pos);
  }

  m_resize_requests[num_seg] = new_size;
  m_resize_requested = true; // TODO Atomic ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
executeResize()
{
  _executeRealloc();

  if (!m_resize_requested) {
    return;
  }
  m_resize_requested = false;

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    if (m_resize_requests[num_seg] == -1) {
      continue;
    }

    const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    if (m_reserved_part_span[num_seg].size() < m_resize_requests[num_seg]) {
      ARCCORE_FATAL("Bad realloc -- New size : {0} -- Needed size : {1}", m_reserved_part_span[num_seg].size(), m_resize_requests[num_seg]);
    }

    m_sizeof_used_part[segment_infos_pos] = m_resize_requests[num_seg];
    m_resize_requests[num_seg] = -1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
executeShrink()
{
  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    if (m_reserved_part_span[num_seg].size() == m_sizeof_used_part[segment_infos_pos]) {
      _requestRealloc(segment_infos_pos);
    }
    else {
      _requestRealloc(segment_infos_pos, m_sizeof_used_part[segment_infos_pos]);
    }
  }
  _executeRealloc();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
_requestRealloc(Int32 owner_pos_segment, Int64 new_capacity) const
{
  m_need_resize[owner_pos_segment] = new_capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
_requestRealloc(Int32 owner_pos_segment) const
{
  m_need_resize[owner_pos_segment] = -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
_executeRealloc()
{
  // Important barrier because everyone must know that we must
  // resize one of the segments we own.
  MPI_Barrier(m_comm_machine);

  // No need for a barrier because MPI_Win_allocate_shared() in _realloc() is
  // blocking.
  _realloc();

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
    m_need_resize[segment_infos_pos] = -1;
  }

  // Important barrier in case an MPI_Win_shared_query() from
  // _reallocCollective() takes too long (another process could
  // call this method and set m_need_resize[m_owner_segment] to
  // true => deadlock in _reallocCollective() on MPI_Win_allocate_shared()
  // due to the continue).
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMultiMachineShMemWinBaseInternal::
_realloc()
{
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "true");

  // Everyone reallocates their segments, if requested.
  for (Integer rank = 0; rank < m_comm_machine_size; ++rank) {
    for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {

      const Int32 local_segment_infos_pos = num_seg + rank * m_nb_segments_per_proc;

      if (m_need_resize[local_segment_infos_pos] == -1)
        continue;

      ARCCORE_ASSERT(m_need_resize[local_segment_infos_pos] >= 0, ("New size must be >= 0"));
      ARCCORE_ASSERT(m_need_resize[local_segment_infos_pos] % m_sizeof_type == 0, ("New size must be % sizeof type"));

      // If we must realloc our segment, we allocate at least a size of m_sizeof_type.
      // If it is not our segment, size 0 so that MPI allocates nothing.
      const Int64 size_seg = (m_comm_machine_rank == rank ? (m_need_resize[local_segment_infos_pos] == 0 ? m_sizeof_type : m_need_resize[local_segment_infos_pos]) : 0);

      // We save the old segment to move the data.
      MPI_Win old_win = m_all_mpi_win[local_segment_infos_pos];
      std::byte* ptr_new_seg = nullptr;

      // If size_seg == 0 then ptr_seg == nullptr.
      int error = MPI_Win_allocate_shared(size_seg, m_sizeof_type, win_info, m_comm_machine, &ptr_new_seg, &m_all_mpi_win[local_segment_infos_pos]);
      if (error != MPI_SUCCESS) {
        MPI_Win_free(&old_win);
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      // We only move the data if it is our segment.
      if (m_comm_machine_rank == rank) {
        // We need two additional pieces of information:
        // - the pointer to the old segment (not possible to retrieve
        //   via m_reserved_part_span due to exchanges),
        // - the size of the new segment (MPI may allocate more than the size
        //   we requested).
        std::byte* ptr_old_seg = nullptr;
        MPI_Aint mpi_reserved_size_new_seg;

        // Old segment.
        {
          MPI_Aint size_old_seg;
          int size_type;
          // Here, ptr_seg is never == nullptr since we always create a
          // segment of at least m_sizeof_type size.
          error = MPI_Win_shared_query(old_win, m_comm_machine_rank, &size_old_seg, &size_type, &ptr_old_seg);
          if (error != MPI_SUCCESS || ptr_old_seg == nullptr) {
            MPI_Win_free(&old_win);
            ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
          }
        }

        // New segment.
        {
          std::byte* ptr_seg = nullptr;
          int size_type;
          // Here, ptr_seg is never == nullptr since we always create a
          // segment of at least m_sizeof_type size.
          error = MPI_Win_shared_query(m_all_mpi_win[local_segment_infos_pos], m_comm_machine_rank, &mpi_reserved_size_new_seg, &size_type, &ptr_seg);
          if (error != MPI_SUCCESS || ptr_seg == nullptr || ptr_seg != ptr_new_seg) {
            MPI_Win_free(&old_win);
            ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
          }
        }

        // If the realloc is a reduction in segment size (user
        // used space, if resize for example), we cannot copy all the old data.
        const Int64 min_size = std::min(m_need_resize[local_segment_infos_pos], m_sizeof_used_part[local_segment_infos_pos]);

        memcpy(ptr_new_seg, ptr_old_seg, min_size);
      }

      MPI_Win_free(&old_win);
    }
  }
  MPI_Info_free(&win_info);

  // We reconstruct the spans of the segments we own.
  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    const Int32 segment_infos_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_all_mpi_win[segment_infos_pos], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_reserved_part_span[num_seg] = Span<std::byte>{ ptr_seg, size_seg };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiMultiMachineShMemWinBaseInternal::
_worldToMachine(Int32 world) const
{
  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == world) {
      return i;
    }
  }
  ARCCORE_FATAL("Rank is not in machine");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiMultiMachineShMemWinBaseInternal::
_machineToWorld(Int32 machine) const
{
  return m_machine_ranks[machine];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
