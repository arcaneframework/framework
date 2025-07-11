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

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMACHINEMEMORYWINDOWBASE_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMachineMemoryWindowBase.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMachineMemoryWindowBase
: public IMachineMemoryWindowBase
{
 public:

  explicit MpiMachineMemoryWindowBase(Integer nb_elem_local_section, Integer sizeof_type, const MPI_Comm& comm, Int32 my_node_rank);

  ~MpiMachineMemoryWindowBase() override;

 public:

  Integer sizeofOneElem() const override;

  Integer sizeSegment() const override;
  Integer sizeSegment(Int32 rank) const override;

  void* data() const override;
  void* data(Int32 rank) const override;

  std::pair<Integer, void*> sizeAndDataSegment() const override;
  std::pair<Integer, void*> sizeAndDataSegment(Int32 rank) const override;

 private:

  Integer m_nb_elem_local;
  MPI_Win m_win;
  MPI_Comm m_comm;
  Int32 m_my_rank;
  Integer m_sizeof_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
