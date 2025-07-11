// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiAllInOneMachineMemoryWindowBase.cc                       (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour un noeud              */
/* de calcul avec MPI. Chaque section de processus contiendra le MPI_Win.    */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiAllInOneMachineMemoryWindowBase.h"

#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiAllInOneMachineMemoryWindowBase::
MpiAllInOneMachineMemoryWindowBase(void* node_window, MPI_Aint offset, const MPI_Comm& comm, Int32 my_node_rank)
: m_node_window(node_window)
, m_nb_elem_local(0)
, m_offset(offset)
, m_comm(comm)
, m_my_rank(my_node_rank)
, m_size_type(0)
{
  m_win = reinterpret_cast<MPI_Win*>(static_cast<char*>(m_node_window) - offset);
  void* ptr_win = nullptr;
  int error = MPI_Win_shared_query(*m_win, m_my_rank, &m_nb_elem_local, &m_size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiAllInOneMachineMemoryWindowBase::
~MpiAllInOneMachineMemoryWindowBase() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiAllInOneMachineMemoryWindowBase::
sizeofOneElem() const
{
  return m_size_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiAllInOneMachineMemoryWindowBase::
sizeSegment() const
{
  return static_cast<Integer>(m_nb_elem_local);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiAllInOneMachineMemoryWindowBase::
sizeSegment(Int32 rank) const
{
  MPI_Aint size_win;
  int size_type;
  void* ptr_win = nullptr;

  int error = MPI_Win_shared_query(*m_win, rank, &size_win, &size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return static_cast<Integer>((size_win - m_offset) / size_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiAllInOneMachineMemoryWindowBase::
data() const
{
  return m_node_window;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiAllInOneMachineMemoryWindowBase::
data(Int32 rank) const
{
  MPI_Aint size_win;
  int size_type;
  void* ptr_win = nullptr;

  int error = MPI_Win_shared_query(*m_win, rank, &size_win, &size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return (static_cast<char*>(ptr_win) + m_offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiAllInOneMachineMemoryWindowBase::
sizeAndDataSegment() const
{
  return sizeAndDataSegment(m_my_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiAllInOneMachineMemoryWindowBase::
sizeAndDataSegment(Int32 rank) const
{
  MPI_Aint size_win;
  int size_type;
  void* ptr_win = nullptr;

  int error = MPI_Win_shared_query(*m_win, rank, &size_win, &size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return { static_cast<Integer>((size_win - m_offset) / size_type), (static_cast<char*>(ptr_win) + m_offset) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
