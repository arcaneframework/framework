// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.cc                                          (C) 2000-2023 */
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

  bool hasRealloc(MemoryAllocationArgs) const final { return false; }
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
    deallocate(args, current_ptr);
    return allocate(args, new_size);
  }
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo mem_info) final
  {
    ARCANE_CHECK_CUDA(_deallocate(mem_info, args));
  }

 protected:

  virtual cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs args) = 0;
  virtual cudaError_t _deallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 public:

  static constexpr Int64 page_size = 4096;

  void initialize()
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_UM_PAGE_ALLOC", true))
      m_page_allocate_level = v.value();
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

  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const
  {
    wanted_capacity = AlignedMemoryAllocator3::adjustedCapacity(args, wanted_capacity, element_size);
    const bool use_test = false;
    if (!use_test)
      return wanted_capacity;
    const bool do_page = m_page_allocate_level > 0;
    // Alloue un multiple de la taille d'une page
    // Comme les transfers de la mémoire unifiée se font par page,
    // cela permet de détecter qu'elle allocation provoque le transfert
    // TODO: vérifier que le début de l'allocation est bien un multiple
    // de la taille de page.
    if (do_page) {
      Int64 orig_capacity = wanted_capacity;
      Int64 new_size = orig_capacity * element_size;
      size_t n = new_size / page_size;
      if ((new_size % page_size) != 0)
        ++n;
      new_size = (n + 1) * page_size;
      wanted_capacity = new_size / element_size;
      if (wanted_capacity < orig_capacity)
        wanted_capacity = orig_capacity;
    }
    return wanted_capacity;
  }

 protected:

  cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs args) override
  {
    const bool do_page = m_page_allocate_level > 0;
    cudaError_t r = do_page ? _allocate_page(ptr, new_size, args) : _allocate_direct(ptr, new_size, args);
    return r;
  }

  cudaError_t _deallocate(AllocatedMemoryInfo mem_info, MemoryAllocationArgs args) override
  {
    void* ptr = mem_info.baseAddress();
    const bool do_trace = m_page_allocate_level >= 2;
    if (do_trace) {
      // Utilise un flux spécifique pour être sur que les affichages ne seront pas mélangés
      // en cas de multi-threading
      std::ostringstream ostr;
      ostr << "FREE_MANAGED=" << ptr << " size=" << mem_info.capacity() << " name=" << args.arrayName();
      if (m_page_allocate_level >= 3) {
        String s = platform::getStackTrace();
        if (m_page_allocate_level >= 4) {
          ostr << " stack=" << s;
        }
        impl::MemoryTracer::notifyMemoryFree(ptr, args.arrayName(), s, 0);
      }
      ostr << "\n";
      std::cout << ostr.str();
    }

    return ::cudaFree(ptr);
  }

  cudaError_t _allocate_direct(void** ptr, size_t new_size, MemoryAllocationArgs args)
  {
    cudaError_t r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
    if (r == cudaSuccess)
      _applyHint(*ptr, new_size, args);
    return r;
  }

  cudaError_t _allocate_page(void** ptr, size_t new_size, MemoryAllocationArgs args)
  {
    auto r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
    void* p = *ptr;

    const bool do_trace = m_page_allocate_level >= 2;
    if (do_trace) {
      // Utilise un flux spécifique pour être sur que les affichages ne seront pas mélangés
      // en cas de multi-threading
      std::ostringstream ostr;
      ostr << "MALLOC_MANAGED=" << p << " size=" << new_size << " name=" << args.arrayName();
      if (m_page_allocate_level >= 3) {
        String s = platform::getStackTrace();
        if (m_page_allocate_level >= 4) {
          ostr << " stack=" << s;
        }
        Span<const std::byte> bytes(reinterpret_cast<std::byte*>(p), new_size);
        impl::MemoryTracer::notifyMemoryAllocation(bytes, args.arrayName(), s, 0);
      }
      ostr << "\n";
      std::cout << ostr.str();
    }

    if (r == cudaSuccess)
      _applyHint(*ptr, new_size, args);

    return r;
  }

  void _applyHint(void* p, size_t new_size, MemoryAllocationArgs args)
  {
    // TODO: regarder comment utiliser une autre device que le device 0.
    // (Peut-être prendre cudaGetDevice ?)
    eMemoryLocationHint hint = args.memoryLocationHint();
    //std::cout << "SET_MEMORY_HINT name=" << args.arrayName() << " size=" << new_size << " hint=" << (int)hint << "\n";
    if (hint == eMemoryLocationHint::MainlyDevice || hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, 0));
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
    }
    if (hint == eMemoryLocationHint::MainlyHost) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      //ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, 0));
    }
    if (hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetReadMostly, 0));
    }
  }

 private:

  //! Strictement positif si on alloue page par page
  Int32 m_page_allocate_level = 0;
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
