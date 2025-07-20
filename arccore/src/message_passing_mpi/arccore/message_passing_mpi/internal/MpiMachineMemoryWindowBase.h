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

#include "arccore/collections/Array.h"
#include "arccore/message_passing/IMachineMemoryWindowBase.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMachineMemoryWindowBase
: public IMachineMemoryWindowBase
{
 public:

  explicit MpiMachineMemoryWindowBase(Integer nb_elem_local_section, Integer sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks);

  ~MpiMachineMemoryWindowBase() override;

 public:

  Integer sizeofOneElem() const override;

  Integer sizeSegment() const override;
  Integer sizeSegment(Int32 rank) const override;

  Integer sizeWindow() const override;

  void* dataSegment() const override;
  void* dataSegment(Int32 rank) const override;

  void* dataWindow() const override;

  std::pair<Integer, void*> sizeAndDataSegment() const override;
  std::pair<Integer, void*> sizeAndDataSegment(Int32 rank) const override;

  std::pair<Integer, void*> sizeAndDataWindow() const override;
  void resizeSegment(Integer new_nb_elem) override;

  ConstArrayView<Int32> machineRanks() const override;

 private:

  MPI_Win m_win;
  void* m_ptr_win;

  MPI_Win m_win_nb_elem_segments;
  ArrayView<Integer> m_nb_elem_segments;

  MPI_Win m_win_sum_nb_elem_segments;
  ArrayView<Integer> m_sum_nb_elem_segments;

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_size;
  Int32 m_comm_machine_rank;
  Int32 m_comm_machine_master_rank;
  bool m_is_master_rank;

  Integer m_sizeof_type;

  ConstArrayView<Int32> m_machine_ranks;
  Integer m_my_rank_index;

  Integer m_max_nb_elem_win;
  Integer m_actual_nb_elem_win;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
