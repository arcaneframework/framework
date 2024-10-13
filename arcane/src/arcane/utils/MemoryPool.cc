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

#include <unordered_map>
#include <map>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MemoryPool::Impl
{
 public:

  explicit Impl(IMemoryPoolAllocator* allocator, const String& name)
  : m_allocator(allocator)
  , m_name(name)
  {
  }

 public:

  void* allocateMemory(size_t size);
  void freeMemory(void* ptr, size_t size);
  void dumpStats(std::ostream& ostr);
  void dumpFreeMap(std::ostream& ostr);

 public:

  IMemoryPoolAllocator* m_allocator = nullptr;
  // Contient une liste de couples (taille_mémoire,pointeur) de mémoire allouée.
  std::unordered_multimap<size_t, void*> m_free_memory_map;
  std::unordered_map<void*, size_t> m_allocated_memory_map;
  std::atomic<size_t> m_total_allocated = 0;
  std::atomic<size_t> m_total_free = 0;
  std::atomic<Int32> m_nb_cached = 0;
  size_t m_max_memory_size_to_pool = 1024 * 64 * 4;
  String m_name;

 private:

  void _freeMemory(void* ptr);
  void _addAllocated(void* ptr, size_t size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryPool::
MemoryPool(IMemoryPoolAllocator* allocator, const String& name)
: m_p(std::make_shared<Impl>(allocator, name))
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

void* MemoryPool::Impl::
allocateMemory(size_t size)
{
  if (m_max_memory_size_to_pool != 0 && size > m_max_memory_size_to_pool)
    return m_allocator->allocateMemory(size);

  auto x = m_free_memory_map.find(size);
  void* ptr = nullptr;
  if (x != m_free_memory_map.end()) {
    ptr = x->second;
    m_free_memory_map.erase(x);
    m_total_free -= size;
    ++m_nb_cached;
  }
  else
    ptr = m_allocator->allocateMemory(size);
  _addAllocated(ptr, size);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryPool::Impl::
freeMemory(void* ptr, size_t size)
{
  if (m_max_memory_size_to_pool != 0 && size > m_max_memory_size_to_pool)
    return m_allocator->freeMemory(ptr, size);

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

void MemoryPool::Impl::
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

void MemoryPool::Impl::
dumpStats(std::ostream& ostr)
{
  ostr << "Stats '" << m_name << "' TotalAllocated=" << m_total_allocated
       << " TotalFree=" << m_total_free
       << " nb_allocated=" << m_allocated_memory_map.size()
       << " nb_free=" << m_free_memory_map.size()
       << " nb_cached=" << m_nb_cached
       << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryPool::Impl::
dumpFreeMap(std::ostream& ostr)
{
  std::map<size_t, Int32> nb_alloc_per_size;
  for (const auto& [key, value] : m_free_memory_map) {
    auto x = nb_alloc_per_size.find(key);
    if (x == nb_alloc_per_size.end())
      nb_alloc_per_size.insert(std::make_pair(key, 1));
    else
      ++x->second;
  }
  ostr << "FreeMap '" << m_name << "\n";
  for (const auto& [key, value] : nb_alloc_per_size)
    ostr << "Map size=" << key << " nb_allocated=" << value << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MemoryPool::allocateMemory(size_t size)
{
  return m_p->allocateMemory(size);
}
void MemoryPool::freeMemory(void* ptr, size_t size)
{
  m_p->freeMemory(ptr, size);
}
void MemoryPool::dumpStats(std::ostream& ostr)
{
  m_p->dumpStats(ostr);
}
void MemoryPool::
dumpFreeMap(std::ostream& ostr)
{
  m_p->dumpFreeMap(ostr);
}
String MemoryPool::name() const
{
  return m_p->m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
