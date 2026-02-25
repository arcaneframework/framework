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

#include "arcane/core/internal/MachineShMemWinMemoryAllocator.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/MachineShMemWinBase.h"

#include "arccore/common/AllocatedMemoryInfo.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinMemoryAllocator::
MachineShMemWinMemoryAllocator(IParallelMng* pm)
: m_pm(pm)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo MachineShMemWinMemoryAllocator::
allocate(MemoryAllocationArgs, Int64 new_size)
{
  // Si la taille est égal à zéro, comme on a une création collective, on doit
  // vérifier si la taille est égal à zéro pour tous.
  // if (m_pm->reduce(MessagePassing::ReduceMax, new_size) <= 0) {
  //   return { nullptr, 0 };
  // }

  constexpr Int64 offset = sizeof(MachineShMemWinBase*);
  const Int64 new_size_with_offset = offset + new_size;

#ifdef ARCANE_DEBUG_ALLOCATOR
  m_pm->traceMng()->debug() << "(1/2) MachineShMemWinMemoryAllocator::allocate"
                            << " -- ptr.size() : " << new_size
                            << " -- offset : " << offset
                            << " -- win_size (offset+ptr.size()) : " << new_size_with_offset;
#endif

  auto* win_ptr = new MachineShMemWinBase(m_pm, new_size_with_offset, 1);

  std::byte* addr_base = win_ptr->segmentView().data();
  std::byte* addr_after_offset = addr_base + offset;

  std::memcpy(addr_base, &win_ptr, offset);

#ifdef ARCANE_DEBUG_ALLOCATOR
  m_pm->traceMng()->debug() << "(2/2) MachineShMemWinMemoryAllocator::allocate"
                            << " -- addr_base : " << addr_base
                            << " -- addr_after_offset : " << addr_after_offset;
#endif

  return { addr_after_offset, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo MachineShMemWinMemoryAllocator::
reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size)
{
  if (current_ptr.baseAddress() == nullptr) {
    return allocate({}, new_size);
  }

  MachineShMemWinBase* win = _windowBase(current_ptr);

  constexpr Int64 offset = sizeof(MachineShMemWinBase*);

  const Int64 new_size_with_offset = offset + new_size;

  const Int64 d_old_size = win->segmentView().size();
  std::byte* d_old_addr_base = win->segmentView().data();

  win->resize(new_size_with_offset);

  std::byte* addr_base = win->segmentView().data();
  std::byte* addr_after_offset = addr_base + offset;

#ifdef ARCANE_DEBUG_ALLOCATOR
  m_pm->traceMng()->debug() << "MachineShMemWinMemoryAllocator::reallocate"
                            << " -- old_size : " << d_old_size
                            << " -- old_addr_base : " << d_old_addr_base
                            << " -- new ptr.size() : " << new_size
                            << " -- offset : " << offset
                            << " -- win_size (offset+ptr.size()) : " << new_size_with_offset
                            << " -- addr_base : " << addr_base
                            << " -- addr_after_offset : " << addr_after_offset;
#endif

  return { addr_after_offset, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinMemoryAllocator::
deallocate(MemoryAllocationArgs, AllocatedMemoryInfo ptr)
{
  // Grâce au allocate(), on est sûr que tout le monde a un nullptr (pas
  // besoin de vérifier avec une réduction).
  if (ptr.baseAddress() == nullptr) {
    return;
  }

  MachineShMemWinBase* win_ptr = _windowBase(ptr);

#ifdef ARCANE_DEBUG_ALLOCATOR
  m_pm->traceMng()->debug() << "MachineShMemWinMemoryAllocator::deallocate"
                            << " -- ptr.size() : " << ptr.size()
                            << " -- win_size (offset+ptr.size()) : " << win_ptr->segmentView().size()
                            << " -- addr_base : " << win_ptr->segmentView().data();
#endif

  delete win_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MachineShMemWinMemoryAllocator::
machineRanks(AllocatedMemoryInfo ptr)
{
  return _windowBase(ptr)->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinMemoryAllocator::
barrier(AllocatedMemoryInfo ptr)
{
  _windowBase(ptr)->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinMemoryAllocator::
segmentView(AllocatedMemoryInfo ptr)
{
  const Span<std::byte> view = _windowBase(ptr)->segmentView();
  constexpr Int64 offset = sizeof(MachineShMemWinBase*);
  return view.subSpan(offset, view.size() - offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinMemoryAllocator::
segmentView(AllocatedMemoryInfo ptr, Int32 rank)
{
  const Span<std::byte> view = _windowBase(ptr)->segmentView(rank);
  constexpr Int64 offset = sizeof(MachineShMemWinBase*);
  return view.subSpan(offset, view.size() - offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinBase* MachineShMemWinMemoryAllocator::
_windowBase(AllocatedMemoryInfo ptr)
{
  constexpr Int64 offset = sizeof(MachineShMemWinBase*);

  std::byte* addr_after_offset = static_cast<std::byte*>(ptr.baseAddress());
  std::byte* addr_base = addr_after_offset - offset;

  MachineShMemWinBase* win_ptr = *reinterpret_cast<MachineShMemWinBase**>(addr_base);

#ifdef ARCANE_DEBUG_ALLOCATOR
  std::cout << "MachineShMemWinMemoryAllocator::_windowBase"
            << " -- ptr.size() : " << ptr.size()
            << " -- offset : " << offset
            << " -- addr_base : " << addr_base
            << " -- addr_after_offset : " << addr_after_offset << std::endl;
#endif

#if 0 //def ARCANE_CHECK
  {
    Int64 size_obj = win_ptr->segmentView().size();
    if (size_obj != offset + ptr.size()) {
      // std::cout << "ERROR MachineShMemWinMemoryAllocator::_windowBase"
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
