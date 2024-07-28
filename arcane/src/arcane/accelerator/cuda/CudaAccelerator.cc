// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.cc                                          (C) 2000-2024 */
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

#include "arcane/accelerator/core/internal/MemoryTracer.h"

#include <iostream>

namespace Arcane::Accelerator::Cuda
{
using namespace Arccore;

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
  ex.explain(std::cerr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour 'Cuda'.
 */
class CudaMemoryAllocatorBase
: public Arccore::AlignedMemoryAllocator3
{
 public:

  CudaMemoryAllocatorBase()
  : AlignedMemoryAllocator3(128)
  {}

  bool hasRealloc(MemoryAllocationArgs) const final { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) final
  {
    void* out = nullptr;
    ARCANE_CHECK_CUDA(_allocate(&out, new_size, args));
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCANE_FATAL("Bad alignment for CUDA allocator: offset={0}", (a % 128));
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) final
  {
    AllocatedMemoryInfo a = allocate(args, new_size);
    ARCANE_CHECK_CUDA(cudaMemcpy(a.baseAddress(), current_ptr.baseAddress(), current_ptr.size(), cudaMemcpyDefault));
    deallocate(args, current_ptr);
    return a;
  }
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo mem_info) final
  {
    ARCANE_CHECK_CUDA_NOTHROW(_deallocate(mem_info, args));
  }

 protected:

  virtual cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs args) = 0;
  virtual cudaError_t _deallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe commune pour gérer l'allocation en mémoire unifiée.
 *
 * Cette classe permet de garantir qu'on alloue la mémoire unifiée sur des
 * multiples de la taille d'une page ce qui permet d'éviter des effets de bord
 * entre les allocations pour les transferts entre l'accélérateur CPU et l'hôte.
 *
 * Par défaut on alloue un multiple de la taille de la page.
 */
class CommonUnifiedMemoryAllocatorWrapper
{
 public:

  CommonUnifiedMemoryAllocatorWrapper()
  : m_page_size(platform::getPageSize())
  {
    if (m_page_size <= 0)
      m_page_size = 4096;
  }

  ~CommonUnifiedMemoryAllocatorWrapper()
  {
    std::cout << "NB_ALLOCATE=" << m_nb_allocate
              << " NB_UNALIGNED=" << m_nb_unaligned_allocate
              << "\n";
  }

 public:

  void initialize()
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_UM_PAGE_ALLOC", true))
      m_page_allocate_level = v.value();
  }

  Int64 adjustedCapacity(Int64 wanted_capacity, Int64 element_size) const
  {
    const bool do_page = m_page_allocate_level > 0;
    if (!do_page)
      return wanted_capacity;
    // Alloue un multiple de la taille d'une page
    // Comme les transfers de la mémoire unifiée se font par page,
    // cela permet de détecter quelles allocations provoquent le transfert
    Int64 orig_capacity = wanted_capacity;
    Int64 new_size = orig_capacity * element_size;
    size_t n = new_size / m_page_size;
    if ((new_size % m_page_size) != 0)
      ++n;
    new_size = (n + 1) * m_page_size;
    wanted_capacity = new_size / element_size;
    if (wanted_capacity < orig_capacity)
      wanted_capacity = orig_capacity;
    return wanted_capacity;
  }

  void doDeallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs args)
  {
    void* ptr = mem_info.baseAddress();
    const bool do_page = m_page_allocate_level > 0;
    if (do_page) {
      uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
      if ((addr % m_page_size) != 0) {
        ++m_nb_unaligned_allocate;
      }
    }
    m_tracer.traceDeallocate(mem_info, args);
  }

  void doAllocate(void* ptr, size_t new_size, MemoryAllocationArgs args)
  {
    ++m_nb_allocate;
    m_tracer.traceAllocate(ptr, new_size, args);
  }

 private:

  Int64 m_page_size = 4096;
  Int32 m_page_allocate_level = 1;
  //! Nombre d'allocations
  std::atomic<Int32> m_nb_allocate = 0;
  //! Nombre d'allocations non alignées
  std::atomic<Int32> m_nb_unaligned_allocate = 0;
  impl::MemoryTracerWrapper m_tracer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 public:

  ~UnifiedMemoryCudaMemoryAllocator()
  {
  }

  void initialize()
  {
    m_wrapper.initialize();
  }

 public:

  void notifyMemoryArgsChanged([[maybe_unused]] MemoryAllocationArgs old_args,
                               MemoryAllocationArgs new_args, AllocatedMemoryInfo ptr) override
  {
    void* p = ptr.baseAddress();
    Int64 s = ptr.capacity();
    if (p && s > 0)
      _applyHint(ptr.baseAddress(), ptr.size(), new_args);
  }

  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const override
  {
    wanted_capacity = AlignedMemoryAllocator3::adjustedCapacity(args, wanted_capacity, element_size);
    return m_wrapper.adjustedCapacity(wanted_capacity, element_size);
  }

 protected:

  cudaError_t _deallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs args) override
  {
    m_wrapper.doDeallocate(mem_info, args);
    void* ptr = mem_info.baseAddress();
    return ::cudaFree(ptr);
  }

  cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs args) override
  {
    auto r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
    void* p = *ptr;
    if (r != cudaSuccess)
      return r;

    m_wrapper.doAllocate(p, new_size, args);

    _applyHint(*ptr, new_size, args);
    return cudaSuccess;
  }

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
    //std::cout << "SET_MEMORY_HINT name=" << args.arrayName() << " size=" << new_size << " hint=" << (int)hint << "\n";
    if (hint == eMemoryLocationHint::MainlyDevice || hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, device_id));
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
    }
    if (hint == eMemoryLocationHint::MainlyHost) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      //ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, 0));
    }
    if (hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetReadMostly, device_id));
    }
  }

 private:

  CommonUnifiedMemoryAllocatorWrapper m_wrapper;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 protected:

  cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    return ::cudaMallocHost(ptr, new_size);
  }
  cudaError_t _deallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs) override
  {
    return ::cudaFreeHost(mem_info.baseAddress());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 protected:

  cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    return ::cudaMalloc(ptr, new_size);
  }
  cudaError_t _deallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs) override
  {
    return ::cudaFree(mem_info.baseAddress());
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
