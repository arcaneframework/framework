// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryPool.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Classe pour gérer une liste de zone allouées.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/MemoryPool.h"

#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryPool::
MemoryPool(IMemoryPoolAllocator* allocator)
: m_allocator(allocator)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryPool::
~MemoryPool()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MemoryPool::
allocateMemory(size_t size)
{
  auto x = m_free_memory_map.find(size);
  void* ptr = nullptr;
  if (x != m_free_memory_map.end()) {
    ptr = x->second;
    m_free_memory_map.erase(x);
    m_total_free -= size;
  }
  else
    ptr = m_allocator->allocateMemory(size);
  _addAllocated(ptr, size);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryPool::
freeMemory(void* ptr, size_t size)
{
  auto x = m_allocated_memory_map.find(ptr);
  if (x == m_allocated_memory_map.end())
    ARCANE_FATAL("pointer {0} is not in the allocated map", ptr);
  size_t allocated_size = x->second;
  if (size != allocated_size)
    ARCANE_FATAL("Incoherent size saved_size={0} arg_size={1}", allocated_size, size);
  m_allocated_memory_map.erase(x);
  m_free_memory_map.insert(std::make_pair(allocated_size, ptr));
  m_total_allocated -= allocated_size;
  m_total_free += allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryPool::
_addAllocated(void* ptr, size_t size)
{
#ifdef ARCANE_CHECK
  if (m_allocated_memory_map.find(ptr) != m_allocated_memory_map.end())
    ARCANE_FATAL("pointer {0} (for size={1}) is already in the allocated map", ptr, size);
#endif
  m_allocated_memory_map.insert(std::make_pair(ptr, size));
  m_total_allocated += size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryPool::
dumpStats()
{
  std::cout << "Stats TotalAllocated=" << m_total_allocated
            << " TotalFree=" << m_total_free
            << " nb_allocated=" << m_allocated_memory_map.size()
            << " nb_free=" << m_free_memory_map.size()
            << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
