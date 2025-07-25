// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineMemoryWindowBaseInternal.h                        (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour un noeud              */
/* de calcul avec MPI. Cette fenêtre sera contigüe pour tous les processus   */
/* d'un même noeud.                                                          */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternal.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBaseInternal::
MpiMachineMemoryWindowBaseInternal(Int64 sizeof_segment, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks)
: m_win()
, m_win_sizeof_segments()
, m_win_sum_sizeof_segments()
, m_comm_machine(comm_machine)
, m_comm_machine_size(comm_machine_size)
, m_comm_machine_rank(comm_machine_rank)
, m_sizeof_type(sizeof_type)
, m_machine_ranks(machine_ranks)
, m_max_sizeof_win(0)
, m_actual_sizeof_win(-1)
{
  // Toutes les fenêtres de cette classe doivent être contigües.
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

  // On alloue la fenêtre principale (qui contiendra les données de l'utilisateur.
  // On ne récupère pas le pointeur vers le segment.
  {
    void* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof_segment, m_sizeof_type, win_info, m_comm_machine, &ptr_seg, &m_win);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
  }

  //--------------------------

  // On alloue la fenêtre qui contiendra la taille de chaque segment de la
  // fenêtre principale.
  {
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_allocate_shared(sizeof(Int64), sizeof(Int64), win_info, m_comm_machine, &ptr_seg, &m_win_sizeof_segments);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
    }
    // On utilise le pointeur vers notre segment pour mettre la taille de
    // notre segment de la fenêtre principale.
    *ptr_seg = sizeof_segment;
  }

  // On alloue la fenêtre qui contiendra la position de chaque segment de la
  // fenêtre principale.
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
    // On crée une vue sur toute la fenêtre contenant les tailles.
    // (la boucle est là uniquement en mode debug pour vérifier qu'on a bien
    // des fenêtres contigües).
    {
      MPI_Aint size_seg;
      int size_type;
      Int64* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_win_sizeof_segments, i, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      if (i == 0) {
        m_sizeof_segments_span = Span<Int64>{ ptr_seg, m_comm_machine_size };
      }

      if (m_sizeof_segments_span.data() + i != ptr_seg) {
        ARCCORE_FATAL("Pb d'adresse de segment");
      }
      if (m_sizeof_segments_span[i] != *ptr_seg) {
        ARCCORE_FATAL("Pb taille de segment");
      }
    }

    // On crée une vue sur toute la fenêtre contenant les positions des segments.
    {
      MPI_Aint size_seg;
      int size_type;
      Int64* ptr_seg = nullptr;
      int error = MPI_Win_shared_query(m_win_sum_sizeof_segments, i, &size_seg, &size_type, &ptr_seg);

      if (error != MPI_SUCCESS) {
        ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
      }
      if (i == 0) {
        m_sum_sizeof_segments_span = Span<Int64>{ ptr_seg, m_comm_machine_size };
      }

      if (m_sum_sizeof_segments_span.data() + i != ptr_seg) {
        ARCCORE_FATAL("Pb d'adresse de segment");
      }
    }
  }
#else
  // On crée une vue sur toute la fenêtre contenant les tailles.
  {
    MPI_Aint size_seg;
    int size_type;
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win_sizeof_segments, 0, &size_seg, &size_type, &ptr_seg);
    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_sizeof_segments_span = Span<Int64>{ ptr_seg, m_comm_machine_size };
  }

  // On crée une vue sur toute la fenêtre contenant les positions des segments.
  {
    MPI_Aint size_seg;
    int size_type;
    Int64* ptr_seg = nullptr;
    int error = MPI_Win_shared_query(m_win_sum_sizeof_segments, 0, &size_seg, &size_type, &ptr_seg);

    if (error != MPI_SUCCESS) {
      ARCCORE_FATAL("Error with MPI_Win_shared_query() call");
    }

    m_sum_sizeof_segments_span = Span<Int64>{ ptr_seg, m_comm_machine_size };
  }
#endif

  //--------------------------

  // Seul le processus 0 doit remplir les positions.
  // Tout le monde calcule la taille de la fenêtre principale.
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

  // La taille actuelle de la fenêtre est sa taille max.
  // Utile en cas de resize.
  m_actual_sizeof_win = m_max_sizeof_win;

  //--------------------------

#ifdef ARCCORE_DEBUG
  Int64 sum = 0;

  for (Int32 i = 0; i < m_comm_machine_size; ++i) {
    // On crée la vue vers la fenêtre principale.
    // (la boucle est là uniquement en mode debug pour vérifier qu'on a bien
    // une fenêtre contigüe).
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
      ARCCORE_FATAL("Pb d'adresse de segment");
    }
    if (size_seg != m_sizeof_segments_span[i]) {
      ARCCORE_FATAL("Pb taille de segment");
    }
    sum += size_seg;
  }
  if (sum != m_max_sizeof_win) {
    ARCCORE_FATAL("Pb taille de window -- Expected : {0} -- Found : {1}", m_max_sizeof_win, sum);
  }
#else
  // On crée la vue vers la fenêtre principale.
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

MpiMachineMemoryWindowBaseInternal::
~MpiMachineMemoryWindowBaseInternal()
{
  MPI_Win_free(&m_win);
  MPI_Win_free(&m_win_sizeof_segments);
  MPI_Win_free(&m_win_sum_sizeof_segments);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MpiMachineMemoryWindowBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiMachineMemoryWindowBaseInternal::
segment() const
{
  const Int64 begin_segment = m_sum_sizeof_segments_span[m_comm_machine_rank];
  const Int64 size_segment = m_sizeof_segments_span[m_comm_machine_rank];

  return m_window_span.subSpan(begin_segment, size_segment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MpiMachineMemoryWindowBaseInternal::
segment(Int32 rank) const
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

Span<std::byte> MpiMachineMemoryWindowBaseInternal::
window() const
{
  return m_window_span;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineMemoryWindowBaseInternal::
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

ConstArrayView<Int32> MpiMachineMemoryWindowBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiMachineMemoryWindowBaseInternal::
barrier() const
{
  MPI_Barrier(m_comm_machine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
