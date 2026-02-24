// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineMemoryWindowBaseInternalCreator.cc                (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* MpiMachineMemoryWindowBaseInternal.                                       */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternalCreator.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternal.h"
#include "arccore/message_passing_mpi/internal/MpiMachineShMemWinBaseInternal.h"
#include "arccore/message_passing_mpi/internal/MpiMultiMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBaseInternalCreator::
MpiMachineMemoryWindowBaseInternalCreator(const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, const MPI_Comm& comm_world, Int32 comm_world_size)
: m_comm_machine(comm_machine)
, m_comm_machine_rank(comm_machine_rank)
, m_comm_machine_size(comm_machine_size)
{
  UniqueArray<Int32> global_ranks(comm_world_size);
  UniqueArray<Int32> machine_ranks(comm_world_size);

  for (Int32 i = 0; i < comm_world_size; ++i) {
    global_ranks[i] = i;
  }
  MPI_Group comm_world_group;
  MPI_Comm_group(comm_world, &comm_world_group);

  MPI_Group machine_comm_group;
  MPI_Comm_group(m_comm_machine, &machine_comm_group);

  MPI_Group_translate_ranks(comm_world_group, comm_world_size, global_ranks.data(), machine_comm_group, machine_ranks.data());

  m_machine_ranks.resize(m_comm_machine_size);

  Int32 iter = 0;
  for (Int32 i = 0; i < comm_world_size; ++i) {
    if (machine_ranks[i] != MPI_UNDEFINED) {
      m_machine_ranks[iter++] = i;
    }
  }
  if (iter != m_comm_machine_size) {
    ARCCORE_FATAL("Error in machine_ranks creation");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBaseInternal* MpiMachineMemoryWindowBaseInternalCreator::
createWindow(Int64 sizeof_segment, Int32 sizeof_type) const
{
  return new MpiMachineMemoryWindowBaseInternal(sizeof_segment, sizeof_type, m_comm_machine, m_comm_machine_rank, m_comm_machine_size, m_machine_ranks);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineShMemWinBaseInternal* MpiMachineMemoryWindowBaseInternalCreator::
createDynamicWindow(Int64 sizeof_segment, Int32 sizeof_type) const
{
  return new MpiMachineShMemWinBaseInternal(sizeof_segment, sizeof_type, m_comm_machine, m_comm_machine_rank, m_comm_machine_size, m_machine_ranks);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMultiMachineShMemWinBaseInternal* MpiMachineMemoryWindowBaseInternalCreator::
createDynamicMultiWindow(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type) const
{
  return new MpiMultiMachineShMemWinBaseInternal(sizeof_segments, nb_segments_per_proc, sizeof_type, m_comm_machine, m_comm_machine_rank, m_comm_machine_size, m_machine_ranks);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiMachineMemoryWindowBaseInternalCreator::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
