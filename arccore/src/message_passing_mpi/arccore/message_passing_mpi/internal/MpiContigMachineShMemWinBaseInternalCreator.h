// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiContigMachineShMemWinBaseInternalCreator.h               (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* MpiContigMachineShMemWinBaseInternal.                                     */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPICONTIGMACHINESHMEMWINBASEINTERNALCREATOR_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPICONTIGMACHINESHMEMWINBASEINTERNALCREATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arccore/message_passing/internal/IContigMachineShMemWinBaseInternal.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiContigMachineShMemWinBaseInternal;
class MpiMachineShMemWinBaseInternal;
class MpiMultiMachineShMemWinBaseInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiContigMachineShMemWinBaseInternalCreator
{
 public:

  explicit MpiContigMachineShMemWinBaseInternalCreator(const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, const MPI_Comm& comm_world, Int32 comm_world_size);

  ~MpiContigMachineShMemWinBaseInternalCreator() = default;

 public:

  MpiContigMachineShMemWinBaseInternal* createWindow(Int64 sizeof_segment, Int32 sizeof_type) const;
  MpiMachineShMemWinBaseInternal* createDynamicWindow(Int64 sizeof_segment, Int32 sizeof_type) const;
  MpiMultiMachineShMemWinBaseInternal* createDynamicMultiWindow(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type) const;

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
