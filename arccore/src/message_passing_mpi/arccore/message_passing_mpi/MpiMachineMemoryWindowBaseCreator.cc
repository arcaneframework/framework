// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMachineMemoryWindowBaseCreator.cc                        (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type MpiMachineMemoryWindowBase. */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseCreator.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineMemoryWindowBaseCreator::
MpiMachineMemoryWindowBaseCreator(const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, const MPI_Comm& comm_world, Int32 comm_world_size)
: m_comm_machine(comm_machine)
, m_comm_machine_rank(comm_machine_rank)
, m_comm_machine_size(comm_machine_size)
{
  UniqueArray<Int32> global_ranks(comm_world_size);
  UniqueArray<Int32> machine_ranks(comm_world_size);

  for (Integer i = 0; i < comm_world_size; ++i) {
    global_ranks[i] = i;
  }
  MPI_Group comm_world_group;
  MPI_Comm_group(comm_world, &comm_world_group);

  MPI_Group machine_comm_group;
  MPI_Comm_group(m_comm_machine, &machine_comm_group);

  MPI_Group_translate_ranks(comm_world_group, comm_world_size, global_ranks.data(), machine_comm_group, machine_ranks.data());

  Int64 final_size = 0;
  for (Int32 rank : machine_ranks) {
    if (rank != MPI_UNDEFINED) {
      final_size++;
    }
  }
  if (final_size != m_comm_machine_size) {
    ARCCORE_FATAL("Pb sizeof comm");
  }
  m_machine_ranks.resize(final_size);

  Int32 iter = 0;
  for (Int32 rank : machine_ranks) {
    if (rank != MPI_UNDEFINED) {
      m_machine_ranks[iter++] = rank;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMachineMemoryWindowBase* MpiMachineMemoryWindowBaseCreator::
createWindow(Integer nb_elem_local_section, Integer sizeof_type) const
{
  return new MpiMachineMemoryWindowBase(nb_elem_local_section, sizeof_type, m_comm_machine, m_comm_machine_rank, m_comm_machine_size, m_machine_ranks);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MpiMachineMemoryWindowBaseCreator::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
