// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineShMemWinBaseInternal.h                 (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMACHINESHMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMACHINESHMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arccore/message_passing/internal/IMachineShMemWinBaseInternal.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMachineShMemWinBaseInternal
: public IMachineShMemWinBaseInternal
{
 public:

  explicit MpiMachineShMemWinBaseInternal(Int64 sizeof_segment, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks);

  ~MpiMachineShMemWinBaseInternal() override;

 public:

  Int32 sizeofOneElem() const override;
  ConstArrayView<Int32> machineRanks() const override;
  void barrier() const override;

  Span<std::byte> segmentView() override;
  Span<std::byte> segmentView(Int32 rank) override;

  Span<const std::byte> segmentConstView() const override;
  Span<const std::byte> segmentConstView(Int32 rank) const override;

  void add(Span<const std::byte> elem) override;
  void add() override;

  void addToAnotherSegment(Int32 rank, Span<const std::byte> elem) override;
  void addToAnotherSegment() override;

  void reserve(Int64 new_capacity) override;
  void reserve() override;

  void resize(Int64 new_size) override;
  void resize() override;

  void shrink() override;

 private:

  void _reallocBarrier(Int64 new_sizeof);
  void _reallocBarrier(Int32 machine_rank, Int64 new_sizeof);
  void _reallocBarrier();
  void _reallocCollective();

  Int32 _worldToMachine(Int32 world) const;
  Int32 _machineToWorld(Int32 machine) const;

 private:

  UniqueArray<MPI_Win> m_all_mpi_win;
  Span<std::byte> m_reserved_part_span;

  MPI_Win m_win_need_resize;
  Span<Int64> m_need_resize;

  MPI_Win m_win_actual_sizeof;
  Span<Int64> m_sizeof_used_part;

  MPI_Win m_win_target_segments;
  Span<Int32> m_target_segments;

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_size = 0;
  Int32 m_comm_machine_rank = 0;

  Int32 m_sizeof_type = 0;

  ConstArrayView<Int32> m_machine_ranks;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
