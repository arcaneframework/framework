// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.cc                                          (C) 2000-2020 */
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

using namespace Arccore;

namespace Arcane::Accelerator::Cuda
{

void arcaneCheckCudaErrors(const TraceInfo& ti,cudaError_t e)
{
  //std::cout << "CUDA TRACE: func=" << ti << "\n";
  if (e!=cudaSuccess){
    //std::cout << "END OF MYVEC1 e=" << e << " v=" << cudaGetErrorString(e) << "\n";
    ARCANE_FATAL("CUDA Error trace={0} e={1} str={2}",ti,e,cudaGetErrorString(e));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur spécifique pour 'Cuda'.
 *
 * Cet allocateur utilise 'cudaMallocManaged' au lieu de 'malloc'
 * pour les allocations.
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

  virtual cudaError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
  }
  virtual cudaError_t _deallocate(void* ptr) override
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

  virtual cudaError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::cudaMallocHost(ptr, new_size);
  }
  virtual cudaError_t _deallocate(void* ptr) override
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

  virtual cudaError_t _allocate(void** ptr, size_t new_size) override
  {
    return ::cudaMalloc(ptr, new_size);
  }
  virtual cudaError_t _deallocate(void* ptr) override
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
