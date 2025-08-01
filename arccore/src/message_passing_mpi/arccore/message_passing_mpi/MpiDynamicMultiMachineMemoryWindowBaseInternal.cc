// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDynamicMultiMachineMemoryWindowBaseInternal.h            (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées. Un processus peut posséder plusieurs segments.      */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiDynamicMultiMachineMemoryWindowBaseInternal.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Le sizeof_segments ne doit pas être conservé !
MpiDynamicMultiMachineMemoryWindowBaseInternal::
MpiDynamicMultiMachineMemoryWindowBaseInternal(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win_need_resize()
, m_win_actual_sizeof()
, m_win_owner_pos_segments()
, m_comm_machine(comm_machine)
, m_comm_machine_size(comm_machine_size)
, m_comm_machine_rank(comm_machine_rank)
, m_sizeof_type(sizeof_type)
, m_nb_segments_per_proc(nb_segments_per_proc)
, m_machine_ranks(machine_ranks)
, m_exchange_requests(std::make_unique<Int32[]>(nb_segments_per_proc * 2))
, m_exchange_requested(false)
, m_add_requests(std::make_unique<Span<const std::byte>[]>(nb_segments_per_proc))
, m_add_requested(false)
, m_resize_requests(std::make_unique<Int64[]>(nb_segments_per_proc))
, m_resize_requested(false)
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

  m_exchange_requests_owner_segment = SmallSpan<Int32>{ m_exchange_requests.get(), nb_segments_per_proc };
  m_exchange_requests_pos_segment = SmallSpan<Int32>{ m_exchange_requests.get() + nb_segments_per_proc, nb_segments_per_proc };

  m_add_requests_span = SmallSpan<Span<const std::byte>>{ m_add_requests.get(), nb_segments_per_proc };

  m_resize_requests_span = SmallSpan<Int64>{ m_resize_requests.get(), nb_segments_per_proc };

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    m_exchange_requests_owner_segment[num_seg] = comm_machine_rank;
    m_exchange_requests_pos_segment[num_seg] = num_seg;

    m_add_requests_span[num_seg] = Span<const std::byte>{ nullptr, 0 };

    m_resize_requests_span[num_seg] = -1;
  }

  MPI_Info win_info_true;
  MPI_Info_create(&win_info_true);
  MPI_Info_set(win_info_true, "alloc_shared_noncontig", "true");

  MPI_Info win_info_false;
  MPI_Info_create(&win_info_false);
  MPI_Info_set(win_info_false, "alloc_shared_noncontig", "false");

  const Int32 pos_my_wins = m_comm_machine_rank * m_nb_segments_per_proc;

  {
    // On crée tous les segments de tous les processus.
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

      // Attention : L'utilisateur demande un nombre minimum d'éléments réservés.
      // Mais MPI réserve la taille qu'il veut (effet du alloc_shared_noncontig=true).
      // On est juste sûr que la taille qu'il a réservée est supérieure ou égale à sizeof_segment.
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
      int error = MPI_Win_allocate_shared(static_cast<Int64>(sizeof(Int32)) * m_nb_segments_per_proc * 2, sizeof(Int32), win_info_false, m_comm_machine, &ptr_seg, &m_win_owner_pos_segments);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      int error = MPI_Win_shared_query(m_win_owner_pos_segments, 0, &size_seg, &size_type, &ptr_win);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }

      m_owner_segments = Span<Int32>{ ptr_win, m_comm_machine_size * m_nb_segments_per_proc };
      m_id_segments = Span<Int32>{ ptr_win + m_comm_machine_size * m_nb_segments_per_proc, m_comm_machine_size * m_nb_segments_per_proc };

      for (Integer i = 0; i < m_nb_segments_per_proc; ++i) {
        //m_owner_pos_segments[i + pos_my_wins] = i + pos_my_wins;
        m_owner_segments[i + pos_my_wins] = m_comm_machine_rank;
        m_id_segments[i + pos_my_wins] = i;
      }
    }
    if (ptr_win + pos_my_wins * 2 != ptr_seg) {
      ARCCORE_FATAL("m_win_owner_segments is noncontig");
    }
  }

  MPI_Info_free(&win_info_false);
  MPI_Info_free(&win_info_true);

  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDynamicMultiMachineMemoryWindowBaseInternal::
~MpiDynamicMultiMachineMemoryWindowBaseInternal()
{
  for (Integer i = 0; i < m_comm_machine_size * m_nb_segments_per_proc; ++i) {
    MPI_Win_free(&m_all_mpi_win[i]);
  }
  MPI_Win_free(&m_win_need_resize);
  MPI_Win_free(&m_win_actual_sizeof);
  MPI_Win_free(&m_win_owner_pos_segments);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiDynamicMultiMachineMemoryWindowBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
barrier() const
{
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiDynamicMultiMachineMemoryWindowBaseInternal::
segment(Int32 num_seg)
{
  const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

  return m_reserved_part_span[num_seg].subSpan(0, m_sizeof_used_part[infos_pos]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiDynamicMultiMachineMemoryWindowBaseInternal::
segment(Int32 rank, Int32 num_seg)
{
  const Int32 owner_id_pos = num_seg + _worldToMachine(rank) * m_nb_segments_per_proc;
  const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

  MPI_Aint size_seg;
  int size_type;
  std::byte* ptr_seg = nullptr;
  int error = MPI_Win_shared_query(m_all_mpi_win[infos_pos], m_owner_segments[owner_id_pos], &size_seg, &size_type, &ptr_seg);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
  }

  return Span<std::byte>{ ptr_seg, m_sizeof_used_part[infos_pos] };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
segmentOwner(Int32 num_seg) const
{
  const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  return _machineToWorld(m_owner_segments[owner_id_pos]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
segmentOwner(Int32 rank, Int32 num_seg) const
{
  const Int32 owner_id_pos = num_seg + _worldToMachine(rank) * m_nb_segments_per_proc;
  return _machineToWorld(m_owner_segments[owner_id_pos]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
segmentPos(Int32 num_seg) const
{
  const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  return m_id_segments[owner_id_pos];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
segmentPos(Int32 rank, Int32 num_seg) const
{
  const Int32 owner_id_pos = num_seg + _worldToMachine(rank) * m_nb_segments_per_proc;
  return m_id_segments[owner_id_pos];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
requestAdd(Int32 num_seg, Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }
  if (elem.empty() || elem.data() == nullptr) {
    return;
  }

  const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

  const Int64 actual_sizeof_win = m_sizeof_used_part[infos_pos];
  const Int64 future_sizeof_win = actual_sizeof_win + elem.size();
  const Int64 old_reserved = m_reserved_part_span[num_seg].size();

  if (future_sizeof_win > old_reserved) {
    _requestRealloc(infos_pos, future_sizeof_win);
  }
  else {
    _requestRealloc(infos_pos);
  }

  m_add_requests_span[num_seg] = elem;
  m_add_requested = true; // TODO Atomic ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
executeAdd()
{
  _executeRealloc();

  if (!m_add_requested) {
    return;
  }
  m_add_requested = false;

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    if (m_add_requests_span[num_seg].empty() || m_add_requests_span[num_seg].data() == nullptr) {
      continue;
    }

    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
    const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

    const Int64 actual_sizeof_win = m_sizeof_used_part[infos_pos];
    const Int64 future_sizeof_win = actual_sizeof_win + m_add_requests_span[num_seg].size();

    if (m_reserved_part_span[num_seg].size() < future_sizeof_win) {
      ARCCORE_FATAL("Bad realloc -- New size : {1} -- Needed size : {2}", m_reserved_part_span[num_seg].size(), future_sizeof_win);
    }

    for (Int64 pos_win = actual_sizeof_win, pos_elem = 0; pos_win < future_sizeof_win; ++pos_win, ++pos_elem) {
      m_reserved_part_span[num_seg][pos_win] = m_add_requests_span[num_seg][pos_elem];
    }
    m_sizeof_used_part[infos_pos] = future_sizeof_win;

    m_add_requests_span[num_seg] = Span<const std::byte>{ nullptr, 0 };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
requestExchangeSegmentWith(Int32 num_seg_src, Int32 rank_dst, Int32 num_seg_dst)
{
  m_exchange_requests_owner_segment[num_seg_src] = _worldToMachine(rank_dst);
  m_exchange_requests_pos_segment[num_seg_src] = num_seg_dst;
  m_exchange_requested = true; // TODO Atomic ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
executeExchangeSegmentWith()
{
  if (!m_exchange_requested) {
    MPI_Barrier(m_comm_machine);
    MPI_Barrier(m_comm_machine);
    return;
  }
  m_exchange_requested = false;

  auto owner_segments_i_have = std::make_unique<Int32[]>(m_nb_segments_per_proc);
  auto pos_segments_i_have = std::make_unique<Int32[]>(m_nb_segments_per_proc);
  auto owner_segments_i_want = std::make_unique<Int32[]>(m_nb_segments_per_proc);
  auto pos_segments_i_want = std::make_unique<Int32[]>(m_nb_segments_per_proc);

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    // La position du segment que l'on possède actuellement.
    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    // La position du segment que l'on souhaiterait avoir.
    const Int32 target_owner_id_pos = m_exchange_requests_pos_segment[num_seg] + m_exchange_requests_owner_segment[num_seg] * m_nb_segments_per_proc;

    // On sauve le segment actuel pour la vérification.
    owner_segments_i_have[num_seg] = m_owner_segments[owner_id_pos];
    pos_segments_i_have[num_seg] = m_id_segments[owner_id_pos];

    // On enregistre le segment que l'on souhaiterait.
    owner_segments_i_want[num_seg] = m_owner_segments[target_owner_id_pos];
    pos_segments_i_want[num_seg] = m_id_segments[target_owner_id_pos];
  }

  MPI_Barrier(m_comm_machine);

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {

    // La position du segment que l'on possède actuellement.
    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    // On le remplace par le segment que l'on souhaite.
    m_owner_segments[owner_id_pos] = owner_segments_i_want[num_seg];
    m_id_segments[owner_id_pos] = pos_segments_i_want[num_seg];
  }

  MPI_Barrier(m_comm_machine);

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    {
      // L'échange étant symétrique, on doit retrouver notre ancien segment à la position du segment que l'on a pris.
      const Int32 target_owner_id_pos = m_exchange_requests_pos_segment[num_seg] + m_exchange_requests_owner_segment[num_seg] * m_nb_segments_per_proc;
      if (m_owner_segments[target_owner_id_pos] != owner_segments_i_have[num_seg] || m_id_segments[target_owner_id_pos] != pos_segments_i_have[num_seg]) {
        ARCCORE_FATAL("Exchange from {0} to {1} is blocked : {1} would like segment {2}:{3}",
                      m_machine_ranks[m_comm_machine_rank], m_exchange_requests_owner_segment[num_seg], m_owner_segments[target_owner_id_pos], m_id_segments[target_owner_id_pos]);
      }
    }

    // Maintenant, on prend la position du segment que l'on possède.
    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    // On cherche la position des informations de ce segment (qui n'ont pas bougé).
    const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

    // (Dans toute cette classe, on utilise "actual_segment_pos_in_array" uniquement pour les tableaux
    // "m_owner_segments" et "m_id_segments". Pour les autres tableaux, on utilise "pos_of_segment_infos").
    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_all_mpi_win[infos_pos], m_owner_segments[owner_id_pos], &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_reserved_part_span[num_seg] = Span<std::byte>{ ptr_seg, size_seg };

    m_exchange_requests_owner_segment[num_seg] = m_comm_machine_rank;
    m_exchange_requests_pos_segment[num_seg] = num_seg;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
resetExchanges()
{
  MPI_Barrier(m_comm_machine);

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;

    m_owner_segments[owner_id_pos] = m_comm_machine_rank;
    m_id_segments[owner_id_pos] = num_seg;

    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_all_mpi_win[owner_id_pos], m_comm_machine_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_reserved_part_span[num_seg] = Span<std::byte>{ ptr_seg, size_seg };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
requestReserve(Int32 num_seg, Int64 new_capacity)
{
  if (new_capacity % m_sizeof_type) {
    ARCCORE_FATAL("new_capacity not valid");
  }

  const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

  if (new_capacity <= m_reserved_part_span[num_seg].size()) {
    _requestRealloc(infos_pos);
    return;
  }
  _requestRealloc(infos_pos, new_capacity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
executeReserve()
{
  _executeRealloc();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
requestResize(Int32 num_seg, Int64 new_size)
{
  if (new_size == -1) {
    return;
  }
  if (new_size < 0 || new_size % m_sizeof_type) {
    ARCCORE_FATAL("new_size not valid");
  }

  const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
  const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

  if (new_size > m_reserved_part_span[num_seg].size()) {
    _requestRealloc(infos_pos, new_size);
  }
  else {
    _requestRealloc(infos_pos);
  }

  m_resize_requests_span[num_seg] = new_size;
  m_resize_requested = true; // TODO Atomic ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
executeResize()
{
  _executeRealloc();

  if (!m_resize_requested) {
    return;
  }
  m_resize_requested = false;

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    if (m_resize_requests_span[num_seg] == -1) {
      continue;
    }

    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
    const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

    if (m_reserved_part_span.size() < m_resize_requests_span[num_seg]) {
      ARCCORE_FATAL("Bad realloc -- New size : {1} -- Needed size : {2}", m_reserved_part_span[num_seg].size(), m_resize_requests_span[num_seg]);
    }

    m_sizeof_used_part[infos_pos] = m_resize_requests_span[num_seg];
    m_resize_requests_span[num_seg] = -1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
executeShrink()
{
  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
    const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

    if (m_reserved_part_span[num_seg].size() == m_sizeof_used_part[infos_pos]) {
      _requestRealloc(infos_pos);
    }
    else {
      _requestRealloc(infos_pos, m_sizeof_used_part[infos_pos]);
    }
  }
  _executeRealloc();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
_requestRealloc(Int32 owner_pos_segment, Int64 new_capacity)
{
  m_need_resize[owner_pos_segment] = new_capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
_requestRealloc(Int32 owner_pos_segment)
{
  m_need_resize[owner_pos_segment] = -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
_executeRealloc()
{
  MPI_Barrier(m_comm_machine);
  _realloc();

  for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {
    const Int32 owner_id_pos = num_seg + m_comm_machine_rank * m_nb_segments_per_proc;
    const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;
    m_need_resize[infos_pos] = -1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDynamicMultiMachineMemoryWindowBaseInternal::
_realloc()
{
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "true");

  for (Integer rank = 0; rank < m_comm_machine_size; ++rank) {
    for (Integer num_seg = 0; num_seg < m_nb_segments_per_proc; ++num_seg) {

      const Int32 owner_id_pos = num_seg + rank * m_nb_segments_per_proc;
      const Int32 infos_pos = m_id_segments[owner_id_pos] + m_owner_segments[owner_id_pos] * m_nb_segments_per_proc;

      if (m_need_resize[owner_id_pos] == -1)
        continue;

      const Int64 size_seg = (m_comm_machine_rank == rank ? (m_need_resize[owner_id_pos] == 0 ? m_sizeof_type : m_need_resize[owner_id_pos]) : 0);

      ARCCORE_ASSERT(m_need_resize[owner_id_pos] >= 0, ("New size must be >= 0"));
      ARCCORE_ASSERT(m_need_resize[owner_id_pos] % m_sizeof_type == 0, ("New size must be % sizeof type"));

      MPI_Win old_win = m_all_mpi_win[owner_id_pos];
      std::byte* ptr_seg = nullptr;

      // Si size_seg == 0 alors ptr_seg == nullptr.
      int error = MPI_Win_allocate_shared(size_seg, m_sizeof_type, win_info, m_comm_machine, &ptr_seg, &m_all_mpi_win[owner_id_pos]);
      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
      }

      if (m_owner_segments[owner_id_pos] == rank) {

        MPI_Aint mpi_reserved_size_seg;
        int size_type;

        // Ici, ptr_seg n'est jamais == nullptr vu que l'on fait toujours un segment d'une taille d'au moins
        // m_sizeof_type.
        error = MPI_Win_shared_query(m_all_mpi_win[infos_pos], m_owner_segments[owner_id_pos], &mpi_reserved_size_seg, &size_type, &ptr_seg);
        if (error != MPI_SUCCESS || ptr_seg == nullptr) {
          ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
        }

        const Int64 min_size = std::min(m_need_resize[owner_id_pos], m_sizeof_used_part[infos_pos]);
        memcpy(ptr_seg, m_reserved_part_span.data(), min_size);

        m_reserved_part_span[m_id_segments[owner_id_pos]] = Span<std::byte>{ ptr_seg, mpi_reserved_size_seg };
      }
      MPI_Win_free(&old_win);
    }
  }
  MPI_Info_free(&win_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
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

Int32 MpiDynamicMultiMachineMemoryWindowBaseInternal::
_machineToWorld(Int32 machine) const
{
  return m_machine_ranks[machine];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
