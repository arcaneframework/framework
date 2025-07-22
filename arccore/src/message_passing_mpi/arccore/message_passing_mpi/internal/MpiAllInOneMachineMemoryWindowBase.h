// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiAllInOneMachineMemoryWindowBase.h                        (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour un noeud              */
/* de calcul avec MPI. Chaque section de processus contiendra le MPI_Win.    */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIALLINONEMACHINEMEMORYWINDOWBASE_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIALLINONEMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiAllInOneMachineMemoryWindowBase
{
 public:

  explicit MpiAllInOneMachineMemoryWindowBase(void* node_window, MPI_Aint offset, const MPI_Comm& comm, Int32 my_node_rank);

  ~MpiAllInOneMachineMemoryWindowBase();

 public:

  Integer sizeofOneElem() const;

  Integer sizeSegment() const;
  Integer sizeSegment(Int32 rank) const;

  void* data() const;
  void* data(Int32 rank) const;

  std::pair<Integer, void*> sizeAndDataSegment() const;
  std::pair<Integer, void*> sizeAndDataSegment(Int32 rank) const;

 private:

  void* m_node_window;
  MPI_Win* m_win;
  MPI_Aint m_nb_elem_local;
  MPI_Aint m_offset;
  MPI_Comm m_comm;
  Int32 m_my_rank;
  Integer m_size_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
