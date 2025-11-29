// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SyclAccelerator.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Backend 'SYCL' pour les accélérateurs.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/sycl/SyclAccelerator.h"
#include "arcane/accelerator/sycl/internal/SyclAcceleratorInternal.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/common/AlignedMemoryAllocator.h"
#include "arccore/common/AllocatedMemoryInfo.h"

#include <iostream>

namespace Arcane::Accelerator::Sycl
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::unique_ptr<sycl::queue> global_default_queue;
namespace
{
  sycl::queue& _defaultQueue()
  {
    return *global_default_queue;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour 'Sycl'.
 */
class SyclMemoryAllocatorBase
: public AlignedMemoryAllocator
{
 public:

  SyclMemoryAllocatorBase()
  : AlignedMemoryAllocator(128)
  {}

  bool hasRealloc(MemoryAllocationArgs) const override { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override
  {
    sycl::queue& q = _defaultQueue();
    void* out = nullptr;
    _allocate(&out, new_size, args, q);
    if (!out)
      ARCCORE_FATAL("Can not allocate memory size={0}", new_size);
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCCORE_FATAL("Bad alignment for SYCL allocator: offset={0}", (a % 128));
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override
  {
    sycl::queue& q = _defaultQueue();
    AllocatedMemoryInfo a = allocate(args, new_size);
    q.submit([&](sycl::handler& cgh) {
      cgh.memcpy(a.baseAddress(), current_ptr.baseAddress(), current_ptr.size());
    });
    q.wait();

    deallocate(args, current_ptr);
    return a;
  }
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override
  {
    sycl::queue& q = _defaultQueue();
    _deallocate(ptr.baseAddress(), args, q);
  }

 protected:

  virtual void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) = 0;
  virtual void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemorySyclMemoryAllocator
: public SyclMemoryAllocatorBase
{
 protected:

  void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) override
  {
    *ptr = sycl::malloc_shared(new_size, q);
  }
  void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) override
  {
    sycl::free(ptr, q);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::UnifiedMemory; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedSyclMemoryAllocator
: public SyclMemoryAllocatorBase
{
 protected:

  void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) override
  {
    // TODO: Faire host-pinned
    *ptr = sycl::malloc_host(new_size, q);
  }
  void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) override
  {
    sycl::free(ptr, q);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceSyclMemoryAllocator
: public SyclMemoryAllocatorBase
{
 protected:

  void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) override
  {
    *ptr = sycl::malloc_device(new_size, q);
  }
  void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) override
  {
    sycl::free(ptr, q);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::Device; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  UnifiedMemorySyclMemoryAllocator unified_memory_sycl_memory_allocator;
  HostPinnedSyclMemoryAllocator host_pinned_sycl_memory_allocator;
  DeviceSyclMemoryAllocator device_sycl_memory_allocator;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

IMemoryAllocator* Sycl::
getSyclMemoryAllocator()
{
  return &unified_memory_sycl_memory_allocator;
}

IMemoryAllocator* Sycl::
getSyclDeviceMemoryAllocator()
{
  return &device_sycl_memory_allocator;
}

IMemoryAllocator* Sycl::
getSyclUnifiedMemoryAllocator()
{
  return &unified_memory_sycl_memory_allocator;
}

IMemoryAllocator* Sycl::
getSyclHostPinnedMemoryAllocator()
{
  return &host_pinned_sycl_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Sycl::
setSyclMemoryQueue(const sycl::queue& memory_queue)
{
  global_default_queue = std::make_unique<sycl::queue>(memory_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
