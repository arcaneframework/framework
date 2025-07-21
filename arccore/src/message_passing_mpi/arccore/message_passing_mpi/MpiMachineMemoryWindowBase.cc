// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineMemoryWindowBase.h                                (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour un noeud              */
/* de calcul avec MPI. Cette fenêtre sera contigüe pour tous les processus   */
/* d'un même noeud.                                                          */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBase.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBase::
MpiMachineMemoryWindowBase(Integer nb_elem_local_section, Integer sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win()
, m_win_nb_elem_segments()
, m_win_sum_nb_elem_segments()
, m_comm_machine(comm_machine)
, m_comm_machine_size(comm_machine_size)
, m_comm_machine_rank(comm_machine_rank)
, m_comm_machine_master_rank(-1)
, m_is_master_rank(false)
, m_sizeof_type(sizeof_type)
, m_machine_ranks(machine_ranks)
, m_my_rank_index(-1)
, m_max_nb_elem_win(0)
, m_actual_nb_elem_win(-1)
{
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

  {
    void* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(nb_elem_local_section * m_sizeof_type, m_sizeof_type, win_info, m_comm_machine, &ptr_seg, &m_win);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
  }
  //--------------------------

  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == m_comm_machine_rank) {
      m_my_rank_index = i;
      break;
    }
  }

  m_comm_machine_master_rank = m_machine_ranks[0];
  m_is_master_rank = (m_comm_machine_master_rank == m_comm_machine_rank);

  //--------------------------

  {
    Integer* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof(Integer), sizeof(Integer), win_info, m_comm_machine, &ptr_seg, &m_win_nb_elem_segments);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
    *ptr_seg = nb_elem_local_section;
  }
  {
    Integer* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof(Integer), sizeof(Integer), win_info, m_comm_machine, &ptr_seg, &m_win_sum_nb_elem_segments);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
  }

  MPI_Info_free(&win_info);

  MPI_Barrier(m_comm_machine);
  //--------------------------

#ifdef ARCCORE_DEBUG
  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    {
      MPI_Aint size_seg;
      int size_type;
      Integer* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_win_nb_elem_segments, m_machine_ranks[i], &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      if (i == 0) {
        m_nb_elem_segments = { m_comm_machine_size, ptr_seg };
      }

      if (m_nb_elem_segments.data() + i != ptr_seg) {
        ARCCORE_FATAL("Pb d'adresse de segment");
      }
      if (m_nb_elem_segments[i] != *ptr_seg) {
        ARCCORE_FATAL("Pb taille de segment");
      }
    }
    {
      MPI_Aint size_seg;
      int size_type;
      Integer* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_win_sum_nb_elem_segments, m_machine_ranks[i], &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      if (i == 0) {
        m_sum_nb_elem_segments = { m_comm_machine_size, ptr_seg };
      }

      if (m_sum_nb_elem_segments.data() + i != ptr_seg) {
        ARCCORE_FATAL("Pb d'adresse de segment");
      }
    }
  }
#else
  {
    MPI_Aint size_seg;
    int size_type;
    Integer* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win_nb_elem_segments, m_comm_machine_master_rank, &size_seg, &size_type, &ptr_seg);
    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_nb_elem_segments = { m_comm_machine_size, ptr_seg };
  }
  {
    MPI_Aint size_seg;
    int size_type;
    Integer* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win_sum_nb_elem_segments, m_comm_machine_master_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_sum_nb_elem_segments = { m_comm_machine_size, ptr_seg };
  }
#endif

  //--------------------------

  if (m_is_master_rank) {
    for (Integer i = 0; i < m_comm_machine_size; ++i) {
      m_sum_nb_elem_segments[i] = m_max_nb_elem_win;
      m_max_nb_elem_win += m_nb_elem_segments[i];
    }
  }
  else {
    for (Integer i = 0; i < m_comm_machine_size; ++i) {
      m_max_nb_elem_win += m_nb_elem_segments[i];
    }
  }

  MPI_Barrier(m_comm_machine);

  m_actual_nb_elem_win = m_max_nb_elem_win;

  //--------------------------

#ifdef ARCCORE_DEBUG
  Integer sum = 0;

  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win, m_machine_ranks[i], &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }
    if (i == 0) {
      m_ptr_win = ptr_seg;
    }

    Integer size_seg2 = static_cast<Integer>(size_seg);

    if (ptr_seg != (m_ptr_win + sum)) {
      ARCCORE_FATAL("Pb d'adresse de segment");
    }
    if (size_seg2 != m_nb_elem_segments[i] * m_sizeof_type) {
      ARCCORE_FATAL("Pb taille de segment");
    }
    sum += size_seg2;
  }
  sum /= m_sizeof_type;
  if (sum != m_max_nb_elem_win) {
    ARCCORE_FATAL("Pb taille de window -- Expected : {0} -- Found : {1}", m_max_nb_elem_win, sum);
  }
#else
  {
    MPI_Aint size_seg;
    int size_type;
    std::byte* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win, m_comm_machine_master_rank, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_ptr_win = ptr_seg;
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBase::
~MpiMachineMemoryWindowBase()
{
  MPI_Win_free(&m_win);
  MPI_Win_free(&m_win_nb_elem_segments);
  MPI_Win_free(&m_win_sum_nb_elem_segments);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiMachineMemoryWindowBase::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiMachineMemoryWindowBase::
sizeSegment() const
{
  return m_nb_elem_segments[m_my_rank_index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiMachineMemoryWindowBase::
sizeSegment(Int32 rank) const
{
  Integer pos = -1;
  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == rank) {
      pos = i;
      break;
    }
  }
  if (pos == -1) {
    ARCCORE_FATAL("Rank is not in machine");
  }

  return m_nb_elem_segments[pos];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiMachineMemoryWindowBase::
sizeWindow() const
{
  return m_actual_nb_elem_win;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiMachineMemoryWindowBase::
dataSegment() const
{
  return (m_ptr_win + (m_sum_nb_elem_segments[m_my_rank_index] * m_sizeof_type));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiMachineMemoryWindowBase::
dataSegment(Int32 rank) const
{
  Integer pos = -1;
  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == rank) {
      pos = i;
      break;
    }
  }
  if (pos == -1) {
    ARCCORE_FATAL("Rank is not in machine");
  }

  return (m_ptr_win + (m_sum_nb_elem_segments[pos] * m_sizeof_type));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiMachineMemoryWindowBase::
dataWindow() const
{
  return m_ptr_win;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiMachineMemoryWindowBase::
sizeAndDataSegment() const
{
  return { (m_nb_elem_segments[m_my_rank_index]), (m_ptr_win + (m_sum_nb_elem_segments[m_my_rank_index] * m_sizeof_type)) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiMachineMemoryWindowBase::
sizeAndDataSegment(Int32 rank) const
{
  Integer pos = -1;
  for (Integer i = 0; i < m_comm_machine_size; ++i) {
    if (m_machine_ranks[i] == rank) {
      pos = i;
      break;
    }
  }
  if (pos == -1) {
    ARCCORE_FATAL("Rank is not in machine");
  }

  return { (m_nb_elem_segments[pos]), (m_ptr_win + (m_sum_nb_elem_segments[pos] * m_sizeof_type)) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiMachineMemoryWindowBase::
sizeAndDataWindow() const
{
  return { m_actual_nb_elem_win, m_ptr_win };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineMemoryWindowBase::
resizeSegment(Integer new_nb_elem)
{
  m_nb_elem_segments[m_my_rank_index] = new_nb_elem;

  MPI_Barrier(m_comm_machine);

  if (m_is_master_rank) {
    Integer sum = 0;
    for (Integer i = 0; i < m_comm_machine_size; ++i) {
      m_sum_nb_elem_segments[i] = sum;
      sum += m_nb_elem_segments[i];
    }
    if (sum > m_max_nb_elem_win) {
      ARCCORE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_nb_elem_win = sum;
  }
  else {
    Integer sum = 0;
    for (Integer i = 0; i < m_comm_machine_size; ++i) {
      sum += m_nb_elem_segments[i];
    }
    if (sum > m_max_nb_elem_win) {
      ARCCORE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_nb_elem_win = sum;
  }
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiMachineMemoryWindowBase::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineMemoryWindowBase::
barrier() const
{
  MPI_Win_sync(m_win);
  MPI_Barrier(m_comm_machine);
  MPI_Win_sync(m_win);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
