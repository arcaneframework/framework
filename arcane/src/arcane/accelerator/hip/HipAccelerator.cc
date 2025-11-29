// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAccelerator.cc                                           (C) 2000-2025 */
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
#include "arcane/utils/IMemoryAllocator.h"

#include "arccore/common/accelerator/internal/AcceleratorMemoryAllocatorBase.h"

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

class ConcreteAllocator
{
 public:

  virtual ~ConcreteAllocator() = default;

 public:

  virtual hipError_t _allocate(void** ptr, size_t new_size) = 0;
  virtual hipError_t _deallocate(void* ptr) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ConcreteAllocatorType>
class UnderlyingAllocator
: public AcceleratorMemoryAllocatorBase::IUnderlyingAllocator
{
 public:

  UnderlyingAllocator() = default;

 public:

  void* allocateMemory(size_t size) final
  {
    void* out = nullptr;
    ARCANE_CHECK_HIP(m_concrete_allocator._allocate(&out, size));
    return out;
  }
  void freeMemory(void* ptr, [[maybe_unused]] size_t size) final
  {
    ARCANE_CHECK_HIP_NOTHROW(m_concrete_allocator._deallocate(ptr));
  }

  void doMemoryCopy(void* destination, const void* source, Int64 size) final
  {
    ARCANE_CHECK_HIP(hipMemcpy(destination, source, size, hipMemcpyDefault));
  }

  eMemoryResource memoryResource() const final
  {
    return m_concrete_allocator.memoryResource();
  }

 public:

  ConcreteAllocatorType m_concrete_allocator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryConcreteAllocator
: public ConcreteAllocator
{
 public:

  hipError_t _deallocate(void* ptr) final
  {
    return ::hipFree(ptr);
  }

  hipError_t _allocate(void** ptr, size_t new_size) final
  {
    auto r = ::hipMallocManaged(ptr, new_size, hipMemAttachGlobal);
    return r;
  }

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::UnifiedMemory; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryHipMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:

  UnifiedMemoryHipMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("UnifiedMemoryHipMemory", new UnderlyingAllocator<UnifiedMemoryConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    _doInitializeUVM();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedConcreteAllocator
: public ConcreteAllocator
{
 public:

  hipError_t _allocate(void** ptr, size_t new_size) final
  {
    return ::hipHostMalloc(ptr, new_size);
  }
  hipError_t _deallocate(void* ptr) final
  {
    return ::hipHostFree(ptr);
  }
  constexpr eMemoryResource memoryResource() const { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedHipMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:
 public:

  HostPinnedHipMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("HostPinnedHipMemory", new UnderlyingAllocator<HostPinnedConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    _doInitializeHostPinned();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceConcreteAllocator
: public ConcreteAllocator
{
 public:

  DeviceConcreteAllocator()
  {
  }

  hipError_t _allocate(void** ptr, size_t new_size) final
  {
    hipError_t r = ::hipMalloc(ptr, new_size);
    return r;
  }
  hipError_t _deallocate(void* ptr) final
  {
    return ::hipFree(ptr);
  }

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::Device; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceHipMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{

 public:

  DeviceHipMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("DeviceHipMemoryAllocator", new UnderlyingAllocator<DeviceConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    _doInitializeDevice();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  UnifiedMemoryHipMemoryAllocator unified_memory_hip_memory_allocator;
  HostPinnedHipMemoryAllocator host_pinned_hip_memory_allocator;
  DeviceHipMemoryAllocator device_hip_memory_allocator;
} // namespace

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

void initializeHipMemoryAllocators()
{
  unified_memory_hip_memory_allocator.initialize();
  device_hip_memory_allocator.initialize();
  host_pinned_hip_memory_allocator.initialize();
}

void finalizeHipMemoryAllocators(ITraceMng* tm)
{
  unified_memory_hip_memory_allocator.finalize(tm);
  device_hip_memory_allocator.finalize(tm);
  host_pinned_hip_memory_allocator.finalize(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
