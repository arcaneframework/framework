// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineShMemWinBaseInternal.h                            (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiMachineShMemWinBaseInternal.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineShMemWinBaseInternal::
MpiMachineShMemWinBaseInternal(Int64 sizeof_segment, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win_need_resize()
, m_win_actual_sizeof()
, m_win_target_segments()
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
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      // Attention : L'utilisateur demande un nombre minimum d'éléments réservés.
      // Mais MPI réserve la taille qu'il veut (effet du alloc_shared_noncontig=true).
      // On est juste sûr que la taille qu'il a réservée est supérieure ou égale à sizeof_segment.
      m_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
    }
  }

  {
    Int64* ptr_seg = nullptr;
    Int64* ptr_win = nullptr;
    {
      int error = MPI_Win_allocate_shared(sizeof(Int64), sizeof(Int64), win_info_false, m_comm_machine, &ptr_seg, &m_win_need_resize);

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

      m_need_resize = Span<Int64>{ ptr_win, m_comm_machine_size };
      m_need_resize[m_comm_machine_rank] = -1;
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
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      m_sizeof_used_part = Span<Int64>{ ptr_win, m_comm_machine_size };
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
      int error = MPI_Win_allocate_shared(sizeof(Int32), sizeof(Int32), win_info_false, m_comm_machine, &ptr_seg, &m_win_target_segments);

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

      m_target_segments = Span<Int32>{ ptr_win, m_comm_machine_size };
      m_target_segments[m_comm_machine_rank] = -1;
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

MpiMachineShMemWinBaseInternal::
~MpiMachineShMemWinBaseInternal()
{
  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    MPI_Win_free(&m_all_mpi_win[i]);
  }
  MPI_Win_free(&m_win_need_resize);
  MPI_Win_free(&m_win_actual_sizeof);
  MPI_Win_free(&m_win_target_segments);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiMachineShMemWinBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiMachineShMemWinBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
barrier() const
{
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiMachineShMemWinBaseInternal::
segmentView()
{
  return m_reserved_part_span.subSpan(0, m_sizeof_used_part[m_comm_machine_rank]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiMachineShMemWinBaseInternal::
segmentView(Int32 rank)
{
  const Int32 machine_rank = _worldToMachine(rank);

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[machine_rank], machine_rank, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
  }

  return Span<std::byte>{ ptr_seg, m_sizeof_used_part[machine_rank] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiMachineShMemWinBaseInternal::
segmentConstView() const
{
  return m_reserved_part_span.subSpan(0, m_sizeof_used_part[m_comm_machine_rank]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MpiMachineShMemWinBaseInternal::
segmentConstView(Int32 rank) const
{
  const Int32 machine_rank = _worldToMachine(rank);

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[machine_rank], machine_rank, &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
  }

  return Span<const std::byte>{ ptr_seg, m_sizeof_used_part[machine_rank] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
add(Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }

  const Int64 actual_sizeof_win = m_sizeof_used_part[m_comm_machine_rank];
  const Int64 future_sizeof_win = actual_sizeof_win + elem.size();
  const Int64 old_reserved = m_reserved_part_span.size();

  if (future_sizeof_win > old_reserved) {
    _reallocBarrier(future_sizeof_win);
    if (m_reserved_part_span.size() < future_sizeof_win) {
      ARCCORE_FATAL("Bad realloc -- Old size : {0} -- New size : {1} -- Needed size : {2}", old_reserved, m_reserved_part_span.size(), future_sizeof_win);
    }
  }
  else {
    _reallocBarrier();
  }

  for (Int64 pos_win = actual_sizeof_win, pos_elem = 0; pos_win < future_sizeof_win; ++pos_win, ++pos_elem) {
    m_reserved_part_span[pos_win] = elem[pos_elem];
  }
  m_sizeof_used_part[m_comm_machine_rank] = future_sizeof_win;

  // Barrière car d'autres peuvent utiliser la taille du segment
  // (m_sizeof_used_part) que nous possédons (segmentView(Int32 rank) par
  // exemple).
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
add()
{
  _reallocBarrier();
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
addToAnotherSegment(Int32 rank, Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }

  const Int32 machine_rank = _worldToMachine(rank);

  m_target_segments[m_comm_machine_rank] = machine_rank;
  MPI_Barrier(m_comm_machine);

  // On doit savoir si quelqu'un va ajouter des éléments dans notre segment
  // pour pouvoir mettre à jour la vue.
  bool is_my_seg_edited = false;
  {
    bool is_found = false;
    for (const Int32 rank_asked : m_target_segments) {
      if (rank_asked == machine_rank) {
        if (!is_found) {
          is_found = true;
        }
        else {
          ARCCORE_FATAL("Two subdomains ask same rank for addToAnotherSegment()");
        }
      }
      if (rank_asked == m_comm_machine_rank) {
        is_my_seg_edited = true;
      }
    }
  }

  Span<std::byte> rank_reserved_part_span;
  {
    MPI_Aint size_seg;
    std::byte* ptr_seg = nullptr;
    int size_type;
    int error = MPI_Win_shared_query(m_all_mpi_win[machine_rank], machine_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }
    rank_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
  }

  const Int64 actual_sizeof_win = m_sizeof_used_part[machine_rank];
  const Int64 future_sizeof_win = actual_sizeof_win + elem.size();
  const Int64 old_reserved = rank_reserved_part_span.size();

  if (future_sizeof_win > old_reserved) {
    _reallocBarrier(machine_rank, future_sizeof_win);

    {
      MPI_Aint size_seg;
      std::byte* ptr_seg = nullptr;
      int size_type;
      int error = MPI_Win_shared_query(m_all_mpi_win[machine_rank], machine_rank, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      rank_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
    }

    if (rank_reserved_part_span.size() < future_sizeof_win) {
      ARCCORE_FATAL("Bad realloc -- Old size : {0} -- New size : {1} -- Needed size : {2}", old_reserved, rank_reserved_part_span.size(), future_sizeof_win);
    }
  }
  else {
    _reallocBarrier();
  }

  for (Int64 pos_win = actual_sizeof_win, pos_elem = 0; pos_win < future_sizeof_win; ++pos_win, ++pos_elem) {
    rank_reserved_part_span[pos_win] = elem[pos_elem];
  }
  m_sizeof_used_part[machine_rank] = future_sizeof_win;

  // Barrière car d'autres peuvent utiliser la taille du segment
  // (m_sizeof_used_part) que nous possédons (segmentView(Int32 machine_rank) par
  // exemple).
  MPI_Barrier(m_comm_machine);
  m_target_segments[m_comm_machine_rank] = -1;

  // On met à jour notre vue.
  if (is_my_seg_edited) {
    MPI_Aint size_seg;
    std::byte* ptr_seg = nullptr;
    int size_type;
    int error = MPI_Win_shared_query(m_all_mpi_win[m_comm_machine_rank], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }
    m_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
addToAnotherSegment()
{
  // Même si on n'ajoute rien, un autre processus pourrait ajouter des
  // éléments dans notre segment.
  MPI_Barrier(m_comm_machine);

  bool is_my_seg_edited = false;
  for (const Int32 rank : m_target_segments) {
    if (rank == m_comm_machine_rank) {
      is_my_seg_edited = true;
      break;
    }
  }

  _reallocBarrier();
  MPI_Barrier(m_comm_machine);

  if (is_my_seg_edited) {
    MPI_Aint size_seg;
    std::byte* ptr_seg = nullptr;
    int size_type;
    int error = MPI_Win_shared_query(m_all_mpi_win[m_comm_machine_rank], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }
    m_reserved_part_span = Span<std::byte>{ ptr_seg, size_seg };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
reserve(Int64 new_capacity)
{
  if (new_capacity <= m_reserved_part_span.size()) {
    _reallocBarrier();
  }
  else {
    _reallocBarrier(new_capacity);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
reserve()
{
  _reallocBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
resize(Int64 new_size)
{
  if (new_size == -1) {
    _reallocBarrier();
    MPI_Barrier(m_comm_machine);
    return;
  }

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
  else {
    _reallocBarrier();
  }
  m_sizeof_used_part[m_comm_machine_rank] = new_size;

  // Barrière car d'autres peuvent utiliser la taille du segment
  // (m_sizeof_used_part) que nous possédons (segmentView(Int32 rank) par
  // exemple).
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
resize()
{
  _reallocBarrier();
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
shrink()
{
  if (m_reserved_part_span.size() == m_sizeof_used_part[m_comm_machine_rank]) {
    _reallocBarrier();
  }
  else {
    _reallocBarrier(m_sizeof_used_part[m_comm_machine_rank]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
_reallocBarrier(Int64 new_sizeof)
{
  m_need_resize[m_comm_machine_rank] = new_sizeof;

  // Barrière importante car tout le monde doit savoir que l'on doit
  // redimensionner le segment que nous possédons.
  MPI_Barrier(m_comm_machine);

  _reallocCollective();

  // Pas besoin de barrière car MPI_Win_allocate_shared() de
  // _reallocCollective() est bloquant.
  m_need_resize[m_comm_machine_rank] = -1;

  // Barrière importante dans le cas où un MPI_Win_shared_query() de
  // _reallocCollective() durerait trop longtemps (un autre processus pourrait
  // rappeler cette méthode et remettre m_need_resize[m_comm_machine_rank] à
  // true => deadlock dans _reallocCollective() sur MPI_Win_allocate_shared()
  // à cause du continue).
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
_reallocBarrier(Int32 machine_rank, Int64 new_sizeof)
{
  m_need_resize[machine_rank] = new_sizeof;

  // Barrière importante car tout le monde doit savoir que l'on doit
  // redimensionner le segment que nous possédons.
  MPI_Barrier(m_comm_machine);

  _reallocCollective();

  // Pas besoin de barrière car MPI_Win_allocate_shared() de
  // _reallocCollective() est bloquant.
  m_need_resize[machine_rank] = -1;

  // Barrière importante dans le cas où un MPI_Win_shared_query() de
  // _reallocCollective() durerait trop longtemps (un autre processus pourrait
  // rappeler cette méthode et remettre m_need_resize[machine_rank] à
  // true => deadlock dans _reallocCollective() sur MPI_Win_allocate_shared()
  // à cause du continue).
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
_reallocBarrier()
{
  MPI_Barrier(m_comm_machine);
  _reallocCollective();
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineShMemWinBaseInternal::
_reallocCollective()
{
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "true");

  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    if (m_need_resize[i] == -1)
      continue;

    ARCCORE_ASSERT(m_need_resize[i] >= 0, ("New size must be >= 0"));
    ARCCORE_ASSERT(m_need_resize[i] % m_sizeof_type == 0, ("New size must be % sizeof type"));

    const Int64 size_seg = (m_comm_machine_rank == i ? (m_need_resize[i] == 0 ? m_sizeof_type : m_need_resize[i]) : 0);

    MPI_Win old_win = m_all_mpi_win[i];

    std::byte* ptr_seg = nullptr;

    // Si size_seg == 0 alors ptr_seg == nullptr.
    int error = MPI_Win_allocate_shared(size_seg, m_sizeof_type, win_info, m_comm_machine, &ptr_seg, &m_all_mpi_win[i]);
    if (error != MPI_SUCCESS) {
      MPI_Info_free(&win_info);
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }

    if (m_comm_machine_rank == i) {

      MPI_Aint mpi_reserved_size_seg;
      int size_type;

      // Ici, ptr_seg n'est jamais == nullptr vu que l'on fait toujours un segment d'une taille d'au moins
      // m_sizeof_type.
      error = MPI_Win_shared_query(m_all_mpi_win[m_comm_machine_rank], m_comm_machine_rank, &mpi_reserved_size_seg, &size_type, &ptr_seg);
      if (error != MPI_SUCCESS || ptr_seg == nullptr) {
        MPI_Win_free(&old_win);
        MPI_Info_free(&win_info);
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      const Int64 min_size = std::min(m_need_resize[i], m_sizeof_used_part[m_comm_machine_rank]);
      memcpy(ptr_seg, m_reserved_part_span.data(), min_size);

      m_reserved_part_span = Span<std::byte>{ ptr_seg, mpi_reserved_size_seg };
    }
    MPI_Win_free(&old_win);
  }
  MPI_Info_free(&win_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiMachineShMemWinBaseInternal::
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

Int32 MpiMachineShMemWinBaseInternal::
_machineToWorld(Int32 machine) const
{
  return m_machine_ranks[machine];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
