// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDynamicMachineMemoryWindowBaseInternal.h                 (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiDynamicMachineMemoryWindowBaseInternal.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDynamicMachineMemoryWindowBaseInternal::
MpiDynamicMachineMemoryWindowBaseInternal(Int64 sizeof_segment, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win_need_resize()
, m_win_actual_sizeof()
, m_win_owner_segments()
, m_owner_segment(comm_machine_rank)
, m_comm_machine(comm_machine)
, m_comm_machine_size(comm_machine_size)
, m_comm_machine_rank(comm_machine_rank)
, m_sizeof_type(sizeof_type)
, m_machine_ranks(machine_ranks)
{
  if (m_sizeof_type <= 0) {
    ARCCORE_FATAL("Invalid sizeof_type");
  }
  if (sizeof_segment < 0 || sizeof_segment % m_sizeof_type != 0) {
    ARCCORE_FATAL("Invalid initial sizeof_segment");
  }
  m_all_mpi_win.resize(m_comm_machine_size);

  MPI_Info win_info_true;
  MPI_Info_create(&win_info_true);
  MPI_Info_set(win_info_true, "alloc_shared_noncontig", "true");

  MPI_Info win_info_false;
  MPI_Info_create(&win_info_false);
  MPI_Info_set(win_info_false, "alloc_shared_noncontig", "false");

  {
    for (Integer i = 0; i < m_comm_machine_size; ++i) {
      Int64 size_seg = 0;
      if (m_comm_machine_rank == i) {
        if (sizeof_segment == 0)
          size_seg = m_sizeof_type;
        else
          size_seg = sizeof_segment;
      }

      std::byte* ptr_seg = nullptr;
      int error = MPI_Win_allocate_shared(size_seg, m_sizeof_type, win_info_true, m_comm_machine, &ptr_seg, &m_all_mpi_win[i]);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      std::byte* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_all_mpi_win[m_comm_machine_rank], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      // Attention : L'utilisateur demande un nombre minimum d'éléments réservés.
      // Mais MPI réserve la taille qu'il veut (effet du alloc_shared_noncontig=true).
      // On est juste sûr que la taille qu'il a réservée est supérieure ou égale à sizeof_segment.
      m_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
    }
  }

  {
    bool* ptr_seg = nullptr;
    bool* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(sizeof(bool), sizeof(bool), win_info_false, m_comm_machine, &ptr_seg, &m_win_need_resize);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_need_resize, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      m_need_resize = Span<bool>{ ptr_win, static_cast<Int64>(sizeof(bool)) * m_comm_machine_size };
      m_need_resize[m_comm_machine_rank] = false;
    }
    if (ptr_win + m_comm_machine_rank != ptr_seg) {
      ARCCORE_FATAL("m_win_need_resize is noncontig");
    }
  }

  {
    Int64* ptr_seg = nullptr;
    Int64* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(sizeof(Int64), sizeof(Int64), win_info_false, m_comm_machine, &ptr_seg, &m_win_actual_sizeof);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_actual_sizeof, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      m_sizeof_used_part = Span<Int64>{ ptr_win, static_cast<Int64>(sizeof(Int64)) * m_comm_machine_size };
      m_sizeof_used_part[m_comm_machine_rank] = sizeof_segment;
    }
    if (ptr_win + m_comm_machine_rank != ptr_seg) {
      ARCCORE_FATAL("m_win_actual_sizeof is noncontig");
    }
  }

  {
    Int32* ptr_seg = nullptr;
    Int32* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(sizeof(Int32), sizeof(Int32), win_info_false, m_comm_machine, &ptr_seg, &m_win_owner_segments);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_owner_segments, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      m_owner_segments = Span<Int32>{ ptr_win, static_cast<Int64>(sizeof(Int32)) * m_comm_machine_size };
      m_owner_segments[m_comm_machine_rank] = m_comm_machine_rank;
    }
    if (ptr_win + m_comm_machine_rank != ptr_seg) {
      ARCCORE_FATAL("m_win_owner_segments is noncontig");
    }
  }

  MPI_Info_free(&win_info_false);
  MPI_Info_free(&win_info_true);

  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDynamicMachineMemoryWindowBaseInternal::
~MpiDynamicMachineMemoryWindowBaseInternal()
{
  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    MPI_Win_free(&m_all_mpi_win[i]);
  }
  MPI_Win_free(&m_win_need_resize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMachineMemoryWindowBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiDynamicMachineMemoryWindowBaseInternal::
segment() const
{
  return m_reserved_part_span.subSpan(0, m_sizeof_used_part[m_owner_segment]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiDynamicMachineMemoryWindowBaseInternal::
segment(Int32 rank) const
{
  Int32 machine_rank = _worldToMachine(rank);

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[machine_rank], machine_rank, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return Span<std::byte>{ ptr_seg, m_sizeof_used_part[machine_rank] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMachineMemoryWindowBaseInternal::
segmentOwner() const
{
  return _machineToWorld(m_owner_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMachineMemoryWindowBaseInternal::
segmentOwner(Int32 rank) const
{
  return _machineToWorld(m_owner_segments[_worldToMachine(rank)]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
add(Span<const std::byte> elem)
{
  if (elem.size() != m_sizeof_type) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }

  const Int64 actual_sizeof_win = m_sizeof_used_part[m_owner_segment];
  const Int64 future_sizeof_win = actual_sizeof_win + m_sizeof_type;
  const Int64 old_reserved = m_reserved_part_span.size();

  if (future_sizeof_win > old_reserved) {
    _reallocBarrier(old_reserved * 2);
    if (m_reserved_part_span.size() < future_sizeof_win) {
      ARCCORE_FATAL("Bad realloc -- Old size : {0} -- New size : {1} -- Needed size : {2}", old_reserved, m_reserved_part_span.size(), future_sizeof_win);
    }
  }

  for (Int64 pos_win = actual_sizeof_win, pos_elem = 0; pos_win < future_sizeof_win; ++pos_win, ++pos_elem) {
    m_reserved_part_span[pos_win] = elem[pos_elem];
  }
  m_sizeof_used_part[m_owner_segment] = future_sizeof_win;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
exchangeSegmentWith(Int32 rank)
{
  const Int32 exchange_with = _worldToMachine(rank);

  if (exchange_with == m_comm_machine_rank) {
    MPI_Barrier(m_comm_machine);
    MPI_Barrier(m_comm_machine);
    return;
  }

  const Int32 segment_i_have = m_owner_segments[m_comm_machine_rank];
  const Int32 segment_i_want = m_owner_segments[exchange_with];

  MPI_Barrier(m_comm_machine);

  m_owner_segments[m_comm_machine_rank] = segment_i_want;

  MPI_Barrier(m_comm_machine);

  if (m_owner_segments[exchange_with] != segment_i_have) {
    ARCCORE_FATAL("Exchange from {0} to {1} is blocked : {1} would like segment {2}",
                  m_machine_ranks[m_comm_machine_rank], rank, m_owner_segments[exchange_with]);
  }

  m_owner_segment = segment_i_want;

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[m_owner_segment], m_owner_segment, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  m_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
exchangeSegmentWith()
{
  MPI_Barrier(m_comm_machine);
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
resetExchanges()
{
  m_owner_segments[m_comm_machine_rank] = m_comm_machine_rank;
  m_owner_segment = m_comm_machine_rank;

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[m_owner_segment], m_owner_segment, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  m_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiDynamicMachineMemoryWindowBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
syncAdd()
{
  MPI_Barrier(m_comm_machine);
  if (_checkNeedRealloc()) {
    _realloc(0);
    syncAdd();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
barrier()
{
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
reserve(Int64 new_capacity)
{
  if (new_capacity <= m_reserved_part_span.size()) {
    MPI_Barrier(m_comm_machine);
    _realloc(0);
    return;
  }
  _reallocBarrier(new_capacity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
reserve()
{
  MPI_Barrier(m_comm_machine);
  _realloc(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
resize(Int64 new_size)
{
  if (new_size < 0 || new_size % m_sizeof_type) {
    ARCCORE_FATAL("new_size not valid");
  }

  Int64 old_reserved = m_reserved_part_span.size();

  if (new_size > old_reserved) {
    _reallocBarrier(new_size);
    if (m_reserved_part_span.size() < new_size) {
      ARCCORE_FATAL("Bad realloc -- Old size : {0} -- New size : {1} -- Needed size : {2}", old_reserved, m_reserved_part_span.size(), new_size);
    }
  }
  MPI_Barrier(m_comm_machine);
  _realloc(0);
  m_sizeof_used_part[m_owner_segment] = new_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
resize()
{
  MPI_Barrier(m_comm_machine);
  _realloc(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
_reallocBarrier(Int64 new_sizeof)
{
  m_need_resize[m_owner_segment] = true;

  // Barrier permettant de lancer le _checkNeedRealloc() des autres procs.
  MPI_Barrier(m_comm_machine);

  _realloc(new_sizeof);

  m_need_resize[m_owner_segment] = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiDynamicMachineMemoryWindowBaseInternal::
_checkNeedRealloc() const
{
  for (const bool elem : m_need_resize) {
    if (elem)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMachineMemoryWindowBaseInternal::
_realloc(Int64 new_sizeof)
{
  ARCCORE_ASSERT(new_sizeof >= 0, ("New size must be >= 0"));
  ARCCORE_ASSERT(new_sizeof % m_sizeof_type == 0, ("New size must be % sizeof type"));

  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "true");

  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    if (!m_need_resize[i])
      continue;

    const Int64 size_seg = (m_comm_machine_rank == i ? (new_sizeof == 0 ? m_sizeof_type : new_sizeof) : 0);

    MPI_Win old_win = m_all_mpi_win[i];

    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(size_seg, m_sizeof_type, win_info, m_comm_machine, &ptr_seg, &m_all_mpi_win[i]);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }

    if (m_owner_segment == i) {
      const Int64 min_size = std::min(new_sizeof, m_sizeof_used_part[m_owner_segment]);
      memcpy(ptr_seg, m_reserved_part_span.data(), min_size);

      MPI_Aint mpi_reserved_size_seg;
      int size_type;
      error = MPI_Win_shared_query(m_all_mpi_win[m_owner_segment], m_owner_segment, &mpi_reserved_size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      m_reserved_part_span = Span<std::byte>{ ptr_seg, mpi_reserved_size_seg };
    }
    MPI_Win_free(&old_win);
  }
  MPI_Info_free(&win_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMachineMemoryWindowBaseInternal::
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

Int32 MpiDynamicMachineMemoryWindowBaseInternal::
_machineToWorld(Int32 machine) const
{
  return m_machine_ranks[machine];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
