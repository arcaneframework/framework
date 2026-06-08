// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMemoryAllocatorBase.h                            (C) 2000-2026 */
/*                                                                           */
/* Base class of a specific allocator for accelerator.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORMEMORYALLOCATORBASE_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORMEMORYALLOCATORBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/common/AllocatedMemoryInfo.h"
#include "arccore/common/AlignedMemoryAllocator.h"
#include "arccore/common/internal/MemoryPool.h"

#include "arccore/common/accelerator/internal/MemoryTracer.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Common class for managing block allocation.
 *
 * This class ensures that memory is allocated in multiples of a block size.
 * This is particularly used for unified memory, which helps avoid side effects
 * between allocations for transfers between the CPU accelerator and the host.
 *
 * By default, it allocates a multiple of 128 bytes.
 */
class ARCCORE_COMMON_EXPORT BlockAllocatorWrapper
{
 public:

  void initialize(Int64 block_size, bool do_block_alloc)
  {
    m_block_size = block_size;
    if (m_block_size <= 0)
      m_block_size = 128;
    m_do_block_allocate = do_block_alloc;
  }

  void dumpStats(std::ostream& ostr, const String& name);

  Int64 adjustedCapacity(Int64 wanted_capacity, Int64 element_size) const
  {
    const bool do_page = m_do_block_allocate;
    if (!do_page)
      return wanted_capacity;
    // Allocates a multiple of the block size
    // For unified memory, the block size is a memory page.
    // Since unified memory transfers happen page by page,
    // this allows detecting which allocations trigger a transfer.
    // We also handle limiting the different block sizes
    // allocated to prevent the eventual MemoryPool from containing too
    // many values.
    Int64 orig_capacity = wanted_capacity;
    Int64 new_size = orig_capacity * element_size;
    Int64 block_size = m_block_size;
    Int64 nb_iter = 4 + (4096 / block_size);
    for (Int64 i = 0; i < nb_iter; ++i) {
      if (new_size >= (4 * block_size))
        block_size *= 4;
      else
        break;
    }
    new_size = _computeNextMultiple(new_size, block_size);
    wanted_capacity = new_size / element_size;
    if (wanted_capacity < orig_capacity)
      wanted_capacity = orig_capacity;
    return wanted_capacity;
  }

  void notifyDoAllocate(void* ptr)
  {
    ++m_nb_allocate;
    if (m_do_block_allocate) {
      uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
      if ((addr % m_block_size) != 0) {
        ++m_nb_unaligned_allocate;
      }
    }
  }

 private:

  //! Block size. Allocation will be a multiple of this size
  Int64 m_block_size = 128;
  //! Indicates whether allocation using \a m_block_size
  bool m_do_block_allocate = true;
  //! Number of allocations
  std::atomic<Int32> m_nb_allocate = 0;
  //! Number of unaligned allocations
  std::atomic<Int32> m_nb_unaligned_allocate = 0;

 private:

  // Calculates the smallest value of \a n multiple of \a multiple
  static Int64 _computeNextMultiple(Int64 n, Int64 multiple)
  {
    Int64 new_n = n / multiple;
    if ((n % multiple) != 0)
      ++new_n;
    return (new_n * multiple);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class of a specific allocator for accelerator.
 */
class ARCCORE_COMMON_EXPORT AcceleratorMemoryAllocatorBase
: public AlignedMemoryAllocator
{
 public:

  using IMemoryPoolAllocator = Arcane::Impl::IMemoryPoolAllocator;
  using BaseClass = AlignedMemoryAllocator;

 public:

  //! List of flags for the memory pool to activate
  enum class MemoryPoolFlags
  {
    UVM = 1,
    Device = 2,
    HostPinned = 4
  };

 public:

  class IUnderlyingAllocator
  : public IMemoryPoolAllocator
  {
   public:

    virtual void doMemoryCopy(void* destination, const void* source, Int64 size) = 0;
    virtual eMemoryResource memoryResource() const = 0;
  };

 public:

  AcceleratorMemoryAllocatorBase(const String& allocator_name, IUnderlyingAllocator* underlying_allocator);

 public:

  void finalize(ITraceMng* tm);

 public:

  bool hasRealloc(MemoryAllocationArgs) const final { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) final
  {
    void* out = m_sub_allocator->allocateMemory(new_size);
    m_block_wrapper.notifyDoAllocate(out);
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCCORE_FATAL("Bad alignment for Accelerator allocator: offset={0}", (a % 128));
    m_tracer.traceAllocate(out, new_size, args);
    _applyHint(out, new_size, args);
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_info, Int64 new_size) final;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo mem_info) final
  {
    void* ptr = mem_info.baseAddress();
    size_t mem_size = mem_info.capacity();
    if (m_use_memory_pool)
      _removeHint(ptr, mem_size, args);
    // Do not throw an exception in case of deallocation errors
    // because they often happen in destructors and cause
    // the code to terminate via std::terminate().
    m_tracer.traceDeallocate(mem_info, args);
    m_sub_allocator->freeMemory(ptr, mem_size);
  }

  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const final
  {
    wanted_capacity = AlignedMemoryAllocator::adjustedCapacity(args, wanted_capacity, element_size);
    return m_block_wrapper.adjustedCapacity(wanted_capacity, element_size);
  }
  eMemoryResource memoryResource() const final { return m_direct_sub_allocator->memoryResource(); }
  void copyMemory([[maybe_unused]] MemoryAllocationArgs args, AllocatedMemoryInfo destination,
                  AllocatedMemoryInfo source) final
  {
    m_direct_sub_allocator->doMemoryCopy(destination.baseAddress(), source.baseAddress(), source.size());
  }
  IMemoryPool* memoryPool() { return &m_memory_pool; }

 protected:

  virtual void _applyHint([[maybe_unused]] void* ptr, [[maybe_unused]] size_t new_size,
                          [[maybe_unused]] MemoryAllocationArgs args) {}
  virtual void _removeHint([[maybe_unused]] void* ptr, [[maybe_unused]] size_t new_size,
                           [[maybe_unused]] MemoryAllocationArgs args) {}

 private:

  Impl::MemoryTracerWrapper m_tracer;
  std::unique_ptr<IUnderlyingAllocator> m_direct_sub_allocator;
  Arcane::Impl::MemoryPool m_memory_pool;
  IMemoryPoolAllocator* m_sub_allocator = nullptr;
  bool m_use_memory_pool = false;
  String m_allocator_name;
  std::atomic<Int32> m_nb_reallocate = 0;
  std::atomic<Int64> m_reallocate_size = 0;
  Int32 m_print_level = 0;
  BlockAllocatorWrapper m_block_wrapper;

 protected:

  //! Initialization for UVM memory
  void _doInitializeUVM(bool default_use_memory_pool = false);
  //! Initialization for HostPinned memory
  void _doInitializeHostPinned(bool default_use_memory_pool = false);
  //! Initialization for Device memory
  void _doInitializeDevice(bool default_use_memory_pool = false);

 protected:

  void _setTraceLevel(Int32 v) { m_tracer.setTraceLevel(v); }

 private:

  // IMPORTANT: must be called before any allocation and must not be modified afterwards.
  void _setUseMemoryPool(bool is_used);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
