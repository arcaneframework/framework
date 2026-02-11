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

#include "arcane/core/DynamicMachineMemoryWindowMemoryAllocator.h"

#include "arcane/core/DynamicMachineMemoryWindowBase.h"
#include "arcane/utils/FatalErrorException.h"
#include "arccore/common/AllocatedMemoryInfo.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMachineMemoryWindowMemoryAllocator::
DynamicMachineMemoryWindowMemoryAllocator(IParallelMng* pm)
: m_pm(pm)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo DynamicMachineMemoryWindowMemoryAllocator::
allocate(MemoryAllocationArgs, Int64 new_size)
{
  Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);
  Int64 new_size_with_offset = offset + new_size;

  auto* win_ptr = new DynamicMachineMemoryWindowBase(m_pm, new_size_with_offset, 1);

  std::byte* addr_base = win_ptr->segmentView().data();
  std::byte* addr_after_offset = addr_base + offset;

  std::memcpy(addr_base, &win_ptr, sizeof(DynamicMachineMemoryWindowBase*));

  return { addr_after_offset, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo DynamicMachineMemoryWindowMemoryAllocator::
reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size)
{
  DynamicMachineMemoryWindowBase* win = _windowBase(current_ptr);

  Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);
  Int64 new_size_with_offset = offset + new_size;

  win->resize(new_size_with_offset);

  std::byte* addr_base = win->segmentView().data();
  std::byte* addr_after_offset = addr_base + offset;

  return { addr_after_offset, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowMemoryAllocator::
deallocate(MemoryAllocationArgs, AllocatedMemoryInfo ptr)
{
  DynamicMachineMemoryWindowBase* win_ptr = _windowBase(ptr);
  delete win_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMachineMemoryWindowBase* DynamicMachineMemoryWindowMemoryAllocator::
_windowBase(AllocatedMemoryInfo ptr)
{
  Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);

  std::byte* addr_after_offset = static_cast<std::byte*>(ptr.baseAddress());
  std::byte* addr_base = addr_after_offset - offset;

  DynamicMachineMemoryWindowBase* win_ptr = *reinterpret_cast<DynamicMachineMemoryWindowBase**>(addr_base);

#ifdef ARCANE_CHECK
  {
    Int64 size_obj = win_ptr->segmentView().size();
    if (size_obj != offset + ptr.size()) {
      ARCANE_FATAL("ptr error");
    }
  }
#endif

  return win_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
