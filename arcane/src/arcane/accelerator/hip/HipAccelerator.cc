// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAccelerator.cc                                           (C) 2000-2021 */
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

using namespace Arccore;

namespace Arcane::Accelerator::Hip
{

void arcaneCheckHipErrors(const TraceInfo& ti,hipError_t e)
{
  //std::cout << "HIP TRACE: func=" << ti << "\n";
  if (e!=hipSuccess){
    //std::cout << "END OF MYVEC1 e=" << e << " v=" << hipGetErrorString(e) << "\n";
    ARCANE_FATAL("HIP Error trace={0} e={1} str={2}",ti,e,hipGetErrorString(e));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour 'Hip'.
 */
class HipMemoryAllocatorBase
: public Arccore::AlignedMemoryAllocator
{
 public:
  HipMemoryAllocatorBase() : AlignedMemoryAllocator(128){}

  bool hasRealloc() const override { return false; }
  void* allocate(size_t new_size) override
  {
    void* out = nullptr;
    ARCANE_CHECK_HIP(_allocate(&out,new_size));
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128)!=0)
      ARCANE_FATAL("Bad alignment for HIP allocator: offset={0}",(a % 128));
    return out;
  }
  void* reallocate(void* current_ptr,size_t new_size) override
  {
    deallocate(current_ptr);
    return allocate(new_size);
  }
  void deallocate(void* ptr) override
  {
    ARCANE_CHECK_HIP(_deallocate(ptr));
  }

 protected:

  virtual hipError_t _allocate(void** ptr, size_t new_size) = 0;
  virtual hipError_t _deallocate(void* ptr) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryHipMemoryAllocator
: public HipMemoryAllocatorBase
{
 protected:

  hipError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::hipMallocManaged(ptr, new_size, hipMemAttachGlobal);
  }
  hipError_t _deallocate(void* ptr) override
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

  hipError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::hipHostMalloc(ptr, new_size);
  }
  hipError_t _deallocate(void* ptr) override
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

  hipError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::hipMalloc(ptr, new_size);
  }
  hipError_t _deallocate(void* ptr) override
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
