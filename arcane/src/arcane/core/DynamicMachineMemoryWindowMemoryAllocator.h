// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowMemoryAllocator.h                 (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe DynamicMachineMemoryWindowBase.    */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWMEMORYALLOCATOR_H
#define ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWMEMORYALLOCATOR_H

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

class DynamicMachineMemoryWindowBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DynamicMachineMemoryWindowMemoryAllocator
: public IMemoryAllocator
{

 public:

  explicit DynamicMachineMemoryWindowMemoryAllocator(IParallelMng* pm);

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

 private:

  DynamicMachineMemoryWindowBase* _windowBase(AllocatedMemoryInfo ptr);

 private:

  IParallelMng* m_pm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
