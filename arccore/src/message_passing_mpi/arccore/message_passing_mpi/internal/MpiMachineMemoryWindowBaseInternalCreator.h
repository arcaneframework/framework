// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineMemoryWindowBaseInternalCreator.h                 (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* MpiMachineMemoryWindowBaseInternal.                                       */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMACHINEMEMORYWINDOWBASEINTERNALCREATOR_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMACHINEMEMORYWINDOWBASEINTERNALCREATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arccore/message_passing/internal/IMachineMemoryWindowBaseInternal.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiMachineMemoryWindowBaseInternal;
class MpiDynamicMachineMemoryWindowBaseInternal;
class MpiDynamicMultiMachineMemoryWindowBaseInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMachineMemoryWindowBaseInternalCreator
{
 public:

  explicit MpiMachineMemoryWindowBaseInternalCreator(const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, const MPI_Comm& comm_world, Int32 comm_world_size);

  ~MpiMachineMemoryWindowBaseInternalCreator() = default;

 public:

  MpiMachineMemoryWindowBaseInternal* createWindow(Int64 sizeof_segment, Int32 sizeof_type) const;
  MpiDynamicMachineMemoryWindowBaseInternal* createDynamicWindow(Int64 sizeof_segment, Int32 sizeof_type) const;
  MpiDynamicMultiMachineMemoryWindowBaseInternal* createDynamicMultiWindow(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type) const;

  ConstArrayView<Int32> machineRanks() const;

 private:

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_rank = 0;
  Int32 m_comm_machine_size = 0;
  UniqueArray<Int32> m_machine_ranks;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
