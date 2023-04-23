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

#include <iostream>

namespace Arcane::Accelerator::Cuda
{
using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
arcaneCheckCudaErrors(const TraceInfo& ti,cudaError_t e)
{
  if (e!=cudaSuccess)
    ARCANE_FATAL("CUDA Error trace={0} e={1} str={2}",ti,e,cudaGetErrorString(e));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
arcaneCheckCudaErrorsNoThrow(const TraceInfo& ti,cudaError_t e)
{
  if (e==cudaSuccess)
    return;
  String str = String::format("CUDA Error trace={0} e={1} str={2}",ti,e,cudaGetErrorString(e));
  FatalErrorException ex(ti,str);
  ex.explain(std::cerr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour 'Cuda'.
 */
class CudaMemoryAllocatorBase
: public Arccore::AlignedMemoryAllocator
{
 public:

  CudaMemoryAllocatorBase()
  : AlignedMemoryAllocator(128)
  {}

  bool hasRealloc() const final { return false; }
  void* allocate(size_t new_size) final
  {
    void* out = nullptr;
    ARCANE_CHECK_CUDA(_allocate(&out, new_size));
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCANE_FATAL("Bad alignment for CUDA allocator: offset={0}", (a % 128));
    return out;
  }
  void* reallocate(void* current_ptr, size_t new_size) final
  {
    deallocate(current_ptr);
    return allocate(new_size);
  }
  void deallocate(void* ptr) final
  {
    ARCANE_CHECK_CUDA(_deallocate(ptr));
  }

 protected:

  virtual cudaError_t _allocate(void** ptr, size_t new_size) = 0;
  virtual cudaError_t _deallocate(void* ptr) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 protected:

  cudaError_t _allocate(void** ptr, size_t new_size) override
  {
    const bool do_page = false;
    if (do_page)
      return _allocate_page(ptr, new_size);
    return ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
  }

  cudaError_t _allocate_page(void** ptr, size_t new_size)
  {
    const size_t page_size = 4096;

    // Alloue un multiple de la taille d'une page
    // Comme les transfers de la mémoire unifiée se font par page,
    // cela permet de détecter qu'elle allocation provoque le transfert
    size_t orig_new_size = new_size;
    size_t n = new_size / page_size;
    if ((new_size % page_size) != 0)
      ++n;
    new_size = (n + 1) * page_size;

    auto r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);

    void* p = *ptr;
    const bool do_trace = false;
    if (do_trace)
      std::cout << "MALLOC_MANAGED=" << p << " size=" << orig_new_size << "\n";

    // Indique qu'on privilégie l'allocation sur le GPU mais que le CPU
    // accédera aussi aux données
    cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    //cudaMemAdvise(p, new_size, cudaMemAdviseSetReadMostly, 0);
    return r;
  }

  cudaError_t _deallocate(void* ptr) override
  {
    return ::cudaFree(ptr);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 protected:

  cudaError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::cudaMallocHost(ptr, new_size);
  }
  cudaError_t _deallocate(void* ptr) override
  {
    return ::cudaFreeHost(ptr);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 protected:

  cudaError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::cudaMalloc(ptr, new_size);
  }
  cudaError_t _deallocate(void* ptr) override
  {
    return ::cudaFree(ptr);
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

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
