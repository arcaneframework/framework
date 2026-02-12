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

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/DynamicMachineMemoryWindowBase.h"

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
  constexpr Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);
  const Int64 new_size_with_offset = offset + new_size;

  auto* win_ptr = new DynamicMachineMemoryWindowBase(m_pm, new_size_with_offset, 1);

  std::byte* addr_base = win_ptr->segmentView().data();
  std::byte* addr_after_offset = addr_base + offset;

  std::memcpy(addr_base, &win_ptr, offset);

  m_pm->traceMng()->debug() << "DynamicMachineMemoryWindowMemoryAllocator::allocate"
                            << " -- ptr.size() : " << new_size
                            << " -- offset : " << offset
                            << " -- win_size (offset+ptr.size()) : " << new_size_with_offset
                            << " -- addr_base : " << addr_base
                            << " -- addr_after_offset : " << addr_after_offset;

  return { addr_after_offset, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo DynamicMachineMemoryWindowMemoryAllocator::
reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size)
{
  DynamicMachineMemoryWindowBase* win = _windowBase(current_ptr);

  constexpr Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);

  const Int64 new_size_with_offset = offset + new_size;

  const Int64 d_old_size = win->segmentView().size();
  std::byte* d_old_addr_base = win->segmentView().data();

  win->resize(new_size_with_offset);

  std::byte* addr_base = win->segmentView().data();
  std::byte* addr_after_offset = addr_base + offset;

  m_pm->traceMng()->debug() << "DynamicMachineMemoryWindowMemoryAllocator::reallocate"
                            << " -- old_size : " << d_old_size
                            << " -- old_addr_base : " << d_old_addr_base
                            << " -- new ptr.size() : " << new_size
                            << " -- offset : " << offset
                            << " -- win_size (offset+ptr.size()) : " << new_size_with_offset
                            << " -- addr_base : " << addr_base
                            << " -- addr_after_offset : " << addr_after_offset;

  return { addr_after_offset, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowMemoryAllocator::
deallocate(MemoryAllocationArgs, AllocatedMemoryInfo ptr)
{
  DynamicMachineMemoryWindowBase* win_ptr = _windowBase(ptr);

  m_pm->traceMng()->debug() << "DynamicMachineMemoryWindowMemoryAllocator::deallocate"
                            << " -- ptr.size() : " << ptr.size()
                            << " -- win_size (offset+ptr.size()) : " << win_ptr->segmentView().size()
                            << " -- addr_base : " << win_ptr->segmentView().data();

  delete win_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> DynamicMachineMemoryWindowMemoryAllocator::
segmentView(AllocatedMemoryInfo ptr)
{
  const Span<std::byte> view = _windowBase(ptr)->segmentView();
  constexpr Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);
  return view.subSpan(offset, view.size() - offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> DynamicMachineMemoryWindowMemoryAllocator::
segmentView(AllocatedMemoryInfo ptr, Int32 rank)
{
  const Span<std::byte> view = _windowBase(ptr)->segmentView(rank);
  constexpr Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);
  return view.subSpan(offset, view.size() - offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> DynamicMachineMemoryWindowMemoryAllocator::
machineRanks(AllocatedMemoryInfo ptr)
{
  return _windowBase(ptr)->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowMemoryAllocator::
barrier(AllocatedMemoryInfo ptr)
{
  _windowBase(ptr)->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMachineMemoryWindowBase* DynamicMachineMemoryWindowMemoryAllocator::
_windowBase(AllocatedMemoryInfo ptr)
{
  constexpr Int64 offset = sizeof(DynamicMachineMemoryWindowBase*);

  std::byte* addr_after_offset = static_cast<std::byte*>(ptr.baseAddress());
  std::byte* addr_base = addr_after_offset - offset;

  DynamicMachineMemoryWindowBase* win_ptr = *reinterpret_cast<DynamicMachineMemoryWindowBase**>(addr_base);

  // std::cout << "DynamicMachineMemoryWindowMemoryAllocator::_windowBase"
  //           << " -- ptr.size() : " << ptr.size()
  //           << " -- offset : " << offset
  //           << " -- addr_base : " << addr_base
  //           << " -- addr_after_offset : " << addr_after_offset << std::endl;

#if 0 //def ARCANE_CHECK
  {
    Int64 size_obj = win_ptr->segmentView().size();
    if (size_obj != offset + ptr.size()) {
      // std::cout << "ERROR DynamicMachineMemoryWindowMemoryAllocator::_windowBase"
      //                           << " -- win_size : " << size_obj << std::endl;
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
