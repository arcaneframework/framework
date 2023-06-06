// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAccelerator.cc                                           (C) 2000-2022 */
/*                                                                           */
/* Backend 'HIP' pour les accélérateurs.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/hip/HipAccelerator.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"

#include <iostream>

namespace Arcane::Accelerator::Hip
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
arcaneCheckHipErrors(const TraceInfo& ti,hipError_t e)
{
  if (e!=hipSuccess){
    ARCANE_FATAL("HIP Error trace={0} e={1} str={2}",ti,e,hipGetErrorString(e));
  }
}

void
arcaneCheckHipErrorsNoThrow(const TraceInfo& ti,hipError_t e)
{
  if (e==hipSuccess)
    return;
  String str = String::format("HIP Error trace={0} e={1} str={2}",ti,e,hipGetErrorString(e));
  FatalErrorException ex(ti,str);
  ex.explain(std::cerr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour 'Hip'.
 */
class HipMemoryAllocatorBase
: public Arccore::AlignedMemoryAllocator3
{
 public:

  HipMemoryAllocatorBase()
  : AlignedMemoryAllocator3(128)
  {}

  bool hasRealloc(MemoryAllocationArgs) const override { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override
  {
    void* out = nullptr;
    ARCANE_CHECK_HIP(_allocate(&out, new_size, args));
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCANE_FATAL("Bad alignment for HIP allocator: offset={0}", (a % 128));
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override
  {
    AllocatedMemoryInfo a = allocate(args, new_size);
    ARCANE_CHECK_HIP(hipMemcpy(a.baseAddress(), current_ptr.baseAddress(), current_ptr.size(), hipMemcpyDefault));
    deallocate(args, current_ptr);
    return a;
  }
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override
  {
    ARCANE_CHECK_HIP(_deallocate(ptr.baseAddress(), args));
  }

 protected:

  virtual hipError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) = 0;
  virtual hipError_t _deallocate(void* ptr, MemoryAllocationArgs) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryHipMemoryAllocator
: public HipMemoryAllocatorBase
{
 protected:

  hipError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    return ::hipMallocManaged(ptr, new_size, hipMemAttachGlobal);
  }
  hipError_t _deallocate(void* ptr, MemoryAllocationArgs) override
  {
    return ::hipFree(ptr);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedHipMemoryAllocator
: public HipMemoryAllocatorBase
{
 protected:

  hipError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    return ::hipHostMalloc(ptr, new_size);
  }
  hipError_t _deallocate(void* ptr, MemoryAllocationArgs) override
  {
    return ::hipHostFree(ptr);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceHipMemoryAllocator
: public HipMemoryAllocatorBase
{
 protected:

  hipError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    return ::hipMalloc(ptr, new_size);
  }
  hipError_t _deallocate(void* ptr, MemoryAllocationArgs) override
  {
    return ::hipFree(ptr);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  UnifiedMemoryHipMemoryAllocator unified_memory_hip_memory_allocator;
  HostPinnedHipMemoryAllocator host_pinned_hip_memory_allocator;
  DeviceHipMemoryAllocator device_hip_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::IMemoryAllocator*
getHipMemoryAllocator()
{
  return &unified_memory_hip_memory_allocator;
}

Arccore::IMemoryAllocator*
getHipDeviceMemoryAllocator()
{
  return &device_hip_memory_allocator;
}

Arccore::IMemoryAllocator*
getHipUnifiedMemoryAllocator()
{
  return &unified_memory_hip_memory_allocator;
}

Arccore::IMemoryAllocator*
getHipHostPinnedMemoryAllocator()
{
  return &host_pinned_hip_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
