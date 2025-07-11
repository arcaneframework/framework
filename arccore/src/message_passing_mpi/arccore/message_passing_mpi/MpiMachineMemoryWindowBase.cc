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
MpiMachineMemoryWindowBase(Integer nb_elem_local_section, Integer sizeof_type, const MPI_Comm& comm, Int32 my_node_rank)
: m_nb_elem_local(nb_elem_local_section)
, m_win()
, m_comm(comm)
, m_my_rank(my_node_rank)
, m_sizeof_type(sizeof_type)
{
  MPI_Info win_info;
  MPI_Info_create(&win_info);
  void* ptr_win = nullptr;

  MPI_Info_set(win_info, "alloc_shared_noncontig", "false");

  MPI_Comm_rank(m_comm, &m_my_rank);

  int error = MPI_Win_allocate_shared(m_nb_elem_local * sizeof_type, sizeof_type, win_info, m_comm, &ptr_win, &m_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  MPI_Info_free(&win_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBase::
~MpiMachineMemoryWindowBase()
{
  MPI_Win_free(&m_win);
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
  return sizeSegment(m_my_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiMachineMemoryWindowBase::
sizeSegment(Int32 rank) const
{
  MPI_Aint size_win;
  int size_type;
  void* ptr_win = nullptr;

  int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return static_cast<Integer>(size_win / size_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiMachineMemoryWindowBase::
data() const
{
  return data(m_my_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MpiMachineMemoryWindowBase::
data(Int32 rank) const
{
  MPI_Aint size_win;
  int size_type;
  void* ptr_win = nullptr;

  int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return ptr_win;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiMachineMemoryWindowBase::
sizeAndDataSegment() const
{
  return sizeAndDataSegment(m_my_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> MpiMachineMemoryWindowBase::
sizeAndDataSegment(Int32 rank) const
{
  MPI_Aint size_win;
  int size_type;
  void* ptr_win = nullptr;

  int error = MPI_Win_shared_query(m_win, rank, &size_win, &size_type, &ptr_win);

  if (error != MPI_SUCCESS) {
    ARCCORE_FATAL("Error with MPI_Win_allocate_shared() call");
  }

  return { static_cast<Integer>(size_win / size_type), ptr_win };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
