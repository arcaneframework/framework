// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinMemoryAllocator.h                            (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe MachineShMemWinBase.               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_MACHINESHMEMWINMEMORYALLOCATOR_H
#define ARCANE_CORE_INTERNAL_MACHINESHMEMWINMEMORYALLOCATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arccore/common/IMemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MachineShMemWinBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT MachineShMemWinMemoryAllocator
: public IMemoryAllocator
{

 public:

  explicit MachineShMemWinMemoryAllocator(IParallelMng* pm);

 public:

  AllocatedMemoryInfo allocate(MemoryAllocationArgs, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs, AllocatedMemoryInfo ptr) override;
  Int64 adjustedCapacity(MemoryAllocationArgs, Int64 wanted_capacity, Int64) const override
  {
    return wanted_capacity;
  }
  size_t guaranteedAlignment(MemoryAllocationArgs) const override
  {
    return 0;
  }

 public:

  static ConstArrayView<Int32> machineRanks(AllocatedMemoryInfo ptr);
  static void barrier(AllocatedMemoryInfo ptr);
  static Span<std::byte> segmentView(AllocatedMemoryInfo ptr);
  static Span<std::byte> segmentView(AllocatedMemoryInfo ptr, Int32 rank);

 private:

  static MachineShMemWinBase* _windowBase(AllocatedMemoryInfo ptr);

 private:

  IParallelMng* m_pm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
