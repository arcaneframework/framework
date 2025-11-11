// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Backend 'CUDA' pour les accélérateurs.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/cuda/CudaAccelerator.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IMemoryAllocator.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/accelerator/core/internal/AcceleratorMemoryAllocatorBase.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Cuda
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void arcaneCheckCudaErrors(const TraceInfo& ti, cudaError_t e)
{
  if (e != cudaSuccess)
    ARCANE_FATAL("CUDA Error trace={0} e={1} str={2}", ti, e, cudaGetErrorString(e));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void arcaneCheckCudaErrorsNoThrow(const TraceInfo& ti, cudaError_t e)
{
  if (e == cudaSuccess)
    return;
  String str = String::format("CUDA Error trace={0} e={1} str={2}", ti, e, cudaGetErrorString(e));
  FatalErrorException ex(ti, str);
  ex.write(std::cerr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConcreteAllocator
{
 public:

  virtual ~ConcreteAllocator() = default;

 public:

  virtual cudaError_t _allocate(void** ptr, size_t new_size) = 0;
  virtual cudaError_t _deallocate(void* ptr) = 0;
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
    ARCANE_CHECK_CUDA(m_concrete_allocator._allocate(&out, size));
    return out;
  }
  void freeMemory(void* ptr, [[maybe_unused]] size_t size) final
  {
    ARCANE_CHECK_CUDA_NOTHROW(m_concrete_allocator._deallocate(ptr));
  }

  void doMemoryCopy(void* destination, const void* source, Int64 size) final
  {
    ARCANE_CHECK_CUDA(cudaMemcpy(destination, source, size, cudaMemcpyDefault));
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

  UnifiedMemoryConcreteAllocator()
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_USE_ALLOC_ATS", true))
      m_use_ats = v.value();
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_MEMORY_HINT_ON_DEVICE", true))
      m_use_hint_as_mainly_device = (v.value() != 0);
  }

  cudaError_t _deallocate(void* ptr) final
  {
    if (m_use_ats) {
      ::free(ptr);
      return cudaSuccess;
    }
    //std::cout << "CUDA_MANAGED_FREE ptr=" << ptr << "\n";
    return ::cudaFree(ptr);
  }

  cudaError_t _allocate(void** ptr, size_t new_size) final
  {
    if (m_use_ats) {
      *ptr = ::aligned_alloc(128, new_size);
    }
    else {
      auto r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
      //std::cout << "CUDA_MANAGED_MALLOC ptr=" << (*ptr) << " size=" << new_size << "\n";
      //if (new_size < 4000)
      //std::cout << "STACK=" << platform::getStackTrace() << "\n";

      if (r != cudaSuccess)
        return r;

      // Si demandé, indique qu'on préfère allouer sur le GPU.
      // NOTE: Dans ce cas, on récupère le device actuel pour positionner la localisation
      // préférée. Dans le cas où on utilise MemoryPool, cette allocation ne sera effectuée
      // qu'une seule fois. Si le device par défaut pour un thread change au cours du calcul
      // il y aura une incohérence. Pour éviter cela, on pourrait faire un cudaMemAdvise()
      // pour chaque allocation (via _applyHint()) mais ces opérations sont assez couteuses
      // et s'il y a beaucoup d'allocation il peut en résulter une perte de performance.
      if (m_use_hint_as_mainly_device) {
        int device_id = 0;
        void* p = *ptr;
        cudaGetDevice(&device_id);
        ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, _getMemoryLocation(device_id)));
        ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, _getMemoryLocation(cudaCpuDeviceId)));
      }
    }

    return cudaSuccess;
  }

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::UnifiedMemory; }

 public:

  bool m_use_ats = false;
  //! Si vrai, par défaut on considère toutes les allocations comme eMemoryLocationHint::MainlyDevice
  bool m_use_hint_as_mainly_device = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur pour la mémoire unifiée.
 *
 * Pour éviter des effets de bord du driver NVIDIA qui effectue les transferts
 * entre le CPU et le GPU par page. on alloue la mémoire par bloc multiple
 * de la taille d'une page.
 */
class UnifiedMemoryCudaMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:
 public:

  UnifiedMemoryCudaMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("UnifiedMemoryCudaMemory", new UnderlyingAllocator<UnifiedMemoryConcreteAllocator>())
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_MALLOC_TRACE", true))
      _setTraceLevel(v.value());
  }

  void initialize()
  {
    _doInitializeUVM();
  }

 public:

  void notifyMemoryArgsChanged([[maybe_unused]] MemoryAllocationArgs old_args,
                               MemoryAllocationArgs new_args, AllocatedMemoryInfo ptr) final
  {
    void* p = ptr.baseAddress();
    Int64 s = ptr.capacity();
    if (p && s > 0)
      _applyHint(ptr.baseAddress(), ptr.size(), new_args);
  }

 protected:

  void _applyHint(void* p, size_t new_size, MemoryAllocationArgs args)
  {
    eMemoryLocationHint hint = args.memoryLocationHint();
    // Utilise le device actif pour positionner le GPU par défaut
    // On ne le fait que si le \a hint le nécessite pour éviter d'appeler
    // cudaGetDevice() à chaque fois.
    int device_id = 0;
    if (hint == eMemoryLocationHint::MainlyDevice || hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      cudaGetDevice(&device_id);
    }
    auto device_memory_location = _getMemoryLocation(device_id);
    auto cpu_memory_location = _getMemoryLocation(cudaCpuDeviceId);

    //std::cout << "SET_MEMORY_HINT name=" << args.arrayName() << " size=" << new_size << " hint=" << (int)hint << "\n";
    if (hint == eMemoryLocationHint::MainlyDevice || hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, device_memory_location));
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, cpu_memory_location));
    }
    if (hint == eMemoryLocationHint::MainlyHost) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, cpu_memory_location));
      //ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, 0));
    }
    if (hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetReadMostly, device_memory_location));
    }
  }
  void _removeHint(void* p, size_t size, MemoryAllocationArgs args)
  {
    eMemoryLocationHint hint = args.memoryLocationHint();
    if (hint == eMemoryLocationHint::None)
      return;
    int device_id = 0;
    ARCANE_CHECK_CUDA(cudaMemAdvise(p, size, cudaMemAdviseUnsetReadMostly, _getMemoryLocation(device_id)));
  }

 private:

  bool m_use_ats = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedConcreteAllocator
: public ConcreteAllocator
{
 public:

  cudaError_t _allocate(void** ptr, size_t new_size) final
  {
    return ::cudaMallocHost(ptr, new_size);
  }
  cudaError_t _deallocate(void* ptr) final
  {
    return ::cudaFreeHost(ptr);
  }
  constexpr eMemoryResource memoryResource() const { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedCudaMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:
 public:

  HostPinnedCudaMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("HostPinnedCudaMemory", new UnderlyingAllocator<HostPinnedConcreteAllocator>())
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
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_USE_ALLOC_ATS", true))
      m_use_ats = v.value();
  }

  cudaError_t _allocate(void** ptr, size_t new_size) final
  {
    if (m_use_ats) {
      // FIXME: it does not work on WIN32
      *ptr = std::aligned_alloc(128, new_size);
      if (*ptr)
        return cudaSuccess;
      return cudaErrorMemoryAllocation;
    }
    cudaError_t r = ::cudaMalloc(ptr, new_size);
    //std::cout << "ALLOCATE_DEVICE ptr=" << (*ptr) << " size=" << new_size << " r=" << (int)r << "\n";
    return r;
  }
  cudaError_t _deallocate(void* ptr) final
  {
    if (m_use_ats) {
      std::free(ptr);
      return cudaSuccess;
    }
    //std::cout << "FREE_DEVICE ptr=" << ptr << "\n";
    return ::cudaFree(ptr);
  }

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::Device; }

 private:

  bool m_use_ats = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceCudaMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{

 public:

  DeviceCudaMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("DeviceCudaMemoryAllocator", new UnderlyingAllocator<DeviceConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    void _doInitializeDevice();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  UnifiedMemoryCudaMemoryAllocator unified_memory_cuda_memory_allocator;
  HostPinnedCudaMemoryAllocator host_pinned_cuda_memory_allocator;
  DeviceCudaMemoryAllocator device_cuda_memory_allocator;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::IMemoryAllocator*
getCudaMemoryAllocator()
{
  return &unified_memory_cuda_memory_allocator;
}

Arccore::IMemoryAllocator*
getCudaDeviceMemoryAllocator()
{
  return &device_cuda_memory_allocator;
}

Arccore::IMemoryAllocator*
getCudaUnifiedMemoryAllocator()
{
  return &unified_memory_cuda_memory_allocator;
}

Arccore::IMemoryAllocator*
getCudaHostPinnedMemoryAllocator()
{
  return &host_pinned_cuda_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void initializeCudaMemoryAllocators()
{
  unified_memory_cuda_memory_allocator.initialize();
  device_cuda_memory_allocator.initialize();
  host_pinned_cuda_memory_allocator.initialize();
}

void finalizeCudaMemoryAllocators(ITraceMng* tm)
{
  unified_memory_cuda_memory_allocator.finalize(tm);
  device_cuda_memory_allocator.finalize(tm);
  host_pinned_cuda_memory_allocator.finalize(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
