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
#include "arccore/message_passing/internal/IMachineMemoryWindowBase.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMachineMemoryWindowBase
: public IMachineMemoryWindowBase
{
 public:

  explicit MpiMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks);

  ~MpiMachineMemoryWindowBase() override;

 public:

  Int32 sizeofOneElem() const override;

  Span<std::byte> segment() const override;
  Span<std::byte> segment(Int32 rank) const override;
  Span<std::byte> window() const override;

  void resizeSegment(Int64 new_sizeof_segment) override;

  ConstArrayView<Int32> machineRanks() const override;

  void barrier() const override;

 private:

  MPI_Win m_win;
  Span<std::byte> m_window_span;

  MPI_Win m_win_sizeof_segments;
  Span<Int64> m_sizeof_segments_span;

  MPI_Win m_win_sum_sizeof_segments;
  Span<Int64> m_sum_sizeof_segments_span;

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_size;
  Int32 m_comm_machine_rank;

  Int32 m_sizeof_type;

  ConstArrayView<Int32> m_machine_ranks;

  Int64 m_max_sizeof_win;
  Int64 m_actual_sizeof_win;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
