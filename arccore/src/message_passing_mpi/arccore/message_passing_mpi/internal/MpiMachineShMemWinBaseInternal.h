// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineShMemWinBaseInternal.h                            (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of memory windows for a computing node.       */
/* The segments of these windows are not contiguous in memory and can        */
/* be resized.                                                               */
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

  //! Table containing all primary windows.
  //!
  //! Reminder: an MPI_Win = a segment.
  UniqueArray<MPI_Win> m_all_mpi_win;
  //! Table with views on the segments. The size of the views corresponds to all
  //! the reserved memory space.
  Span<std::byte> m_reserved_part_span;

  //! Contiguous window with resize size (or -1 if
  //! resizing is not requested).
  MPI_Win m_win_need_resize;
  //! Global view on contiguous window with resize size (or -1 if
  //! resizing is not requested).
  Span<Int64> m_need_resize;

  //! Contiguous window with the size of the primary windows.
  MPI_Win m_win_actual_sizeof;
  //! Global view on contiguous window with the size of the primary windows.
  Span<Int64> m_sizeof_used_part;

  //! Contiguous window with a request to modify a segment from another
  //! subdomain.
  MPI_Win m_win_target_segments;
  //! Global view on contiguous window with a request to modify a segment
  //! from another subdomain.
  //!
  //! Considering that an owner of segment SD wants to modify segment ST,
  //! they will perform this modification:
  //!
  //! m_target_segments[ST] = SD
  //! (See method \a addToAnotherSegment()).
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
