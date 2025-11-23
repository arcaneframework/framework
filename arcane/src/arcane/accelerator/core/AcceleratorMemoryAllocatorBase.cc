// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMemoryAllocatorBase.cc                           (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un allocateur spécifique pour accélérateur.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/AcceleratorMemoryAllocatorBase.h"

#include "arccore/base/Convert.h"
#include "arccore/trace/ITraceMng.h"

#include "arcane/utils/PlatformUtils.h"

#include <iostream>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockAllocatorWrapper::
dumpStats(std::ostream& ostr, const String& name)
{
  ostr << "Allocator '" << name << "' : nb_allocate=" << m_nb_allocate
       << " nb_unaligned=" << m_nb_unaligned_allocate
       << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorMemoryAllocatorBase::
AcceleratorMemoryAllocatorBase(const String& allocator_name, IUnderlyingAllocator* underlying_allocator)
: AlignedMemoryAllocator3(128)
, m_direct_sub_allocator(underlying_allocator)
, m_memory_pool(m_direct_sub_allocator.get(), allocator_name)
, m_sub_allocator(m_direct_sub_allocator.get())
, m_allocator_name(allocator_name)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_PRINT_LEVEL", true))
    m_print_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorMemoryAllocatorBase::
finalize(ITraceMng* tm)
{
  if (m_print_level >= 1) {
    std::ostringstream ostr;
    if (m_use_memory_pool) {
      m_memory_pool.dumpStats(ostr);
      m_memory_pool.dumpFreeMap(ostr);
    }
    ostr << "Allocator '" << m_allocator_name << "' nb_realloc=" << m_nb_reallocate
         << " realloc_copy=" << m_reallocate_size << "\n";
    m_block_wrapper.dumpStats(ostr, m_allocator_name);
    if (tm)
      tm->info() << ostr.str();
    else
      std::cout << ostr.str();
  }

  m_memory_pool.freeCachedMemory();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo AcceleratorMemoryAllocatorBase::
reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_info, Int64 new_size)
{
  ++m_nb_reallocate;
  Int64 current_size = current_info.size();
  m_reallocate_size += current_size;
  String array_name = args.arrayName();
  const bool do_print = (m_print_level >= 2);
  if (do_print) {
    std::cout << "Reallocate allocator=" << m_allocator_name
              << " current_size=" << current_size
              << " current_capacity=" << current_info.capacity()
              << " new_capacity=" << new_size
              << " ptr=" << current_info.baseAddress();
    if (array_name.null() && m_print_level >= 3) {
      std::cout << " stack=" << platform::getStackTrace();
    }
    else {
      std::cout << " name=" << array_name;
      if (m_print_level >= 4)
        std::cout << " stack=" << platform::getStackTrace();
    }
    std::cout << "\n";
  }
  if (m_use_memory_pool)
    _removeHint(current_info.baseAddress(), current_size, args);
  AllocatedMemoryInfo a = allocate(args, new_size);
  // TODO: supprimer les Hint après le deallocate car la zone mémoire peut être réutilisée.
  m_direct_sub_allocator->doMemoryCopy(a.baseAddress(), current_info.baseAddress(), current_size);
  deallocate(args, current_info);
  return a;
}

//! Initialisation pour la mémoire UVM
void AcceleratorMemoryAllocatorBase::
_doInitializeUVM(bool default_use_memory_pool)
{
  bool do_page_allocate = true;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_UM_PAGE_ALLOC", true))
    do_page_allocate = (v.value() != 0);
  Int64 page_size = platform::getPageSize();
  m_block_wrapper.initialize(page_size, do_page_allocate);

  bool use_memory_pool = default_use_memory_pool;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL", true))
    use_memory_pool = (v.value() & static_cast<int>(MemoryPoolFlags::UVM)) != 0;
  _setUseMemoryPool(use_memory_pool);
}

//! Initialisation pour la mémoire HostPinned
void AcceleratorMemoryAllocatorBase::
_doInitializeHostPinned(bool default_use_memory_pool)
{
  bool use_memory_pool = default_use_memory_pool;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL", true))
    use_memory_pool = (v.value() & static_cast<int>(MemoryPoolFlags::HostPinned)) != 0;
  _setUseMemoryPool(use_memory_pool);
  m_block_wrapper.initialize(128, use_memory_pool);
}

//! Initialisation pour la mémoire Device
void AcceleratorMemoryAllocatorBase::
_doInitializeDevice(bool default_use_memory_pool)
{
  bool use_memory_pool = default_use_memory_pool;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL", true))
    use_memory_pool = (v.value() & static_cast<int>(MemoryPoolFlags::Device)) != 0;
  _setUseMemoryPool(use_memory_pool);
  m_block_wrapper.initialize(128, use_memory_pool);
}

// IMPORTANT: doit être appelé avant toute allocation et ne plus être modifié ensuite.
void AcceleratorMemoryAllocatorBase::
_setUseMemoryPool(bool is_used)
{
  IMemoryPoolAllocator* mem_pool = &m_memory_pool;
  IMemoryPoolAllocator* direct = m_direct_sub_allocator.get();
  m_sub_allocator = (is_used) ? mem_pool : direct;
  m_use_memory_pool = is_used;
  if (is_used) {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL_MAX_BLOCK_SIZE", true)) {
      if (v.value() < 0)
        ARCANE_FATAL("Invalid value '{0}' for memory pool max block size");
      size_t block_size = static_cast<size_t>(v.value());
      m_memory_pool.setMaxCachedBlockSize(block_size);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

 /*---------------------------------------------------------------------------*/
 /*---------------------------------------------------------------------------*/
