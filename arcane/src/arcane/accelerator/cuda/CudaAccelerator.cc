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
: public Arccore::AlignedMemoryAllocator2
{
 public:

  CudaMemoryAllocatorBase()
  : AlignedMemoryAllocator2(128)
  {}

  bool hasRealloc(MemoryAllocationArgs) const final { return false; }
  void* allocate(size_t new_size, MemoryAllocationArgs args) final
  {
    void* out = nullptr;
    ARCANE_CHECK_CUDA(_allocate(&out, new_size, args));
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCANE_FATAL("Bad alignment for CUDA allocator: offset={0}", (a % 128));
    return out;
  }
  void* reallocate(void* current_ptr, size_t new_size, MemoryAllocationArgs args) final
  {
    deallocate(current_ptr, args);
    return allocate(new_size, args);
  }
  void deallocate(void* ptr, MemoryAllocationArgs args) final
  {
    ARCANE_CHECK_CUDA(_deallocate(ptr, args));
  }

 protected:

  virtual cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs args) = 0;
  virtual cudaError_t _deallocate(void* ptr, MemoryAllocationArgs args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 public:

  void initialize()
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_UM_PAGE_ALLOC", true))
      m_page_allocate_level = v.value();
  }

 protected:

  cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    const bool do_page = m_page_allocate_level > 0;
    if (do_page)
      return _allocate_page(ptr, new_size);
    return ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
  }

  cudaError_t _deallocate(void* ptr, MemoryAllocationArgs) override
  {
    const bool do_trace = m_page_allocate_level >= 2;
    if (do_trace) {
      // Utilise un flux spécifique pour être sur que les affichages ne seront pas mélangés
      // en cas de multi-threading
      std::ostringstream ostr;
      ostr << "FREE_MANAGED=" << ptr;
      if (m_page_allocate_level >= 3) {
        String s = platform::getStackTrace();
        ostr << " stack=" << s;
      }
      ostr << "\n";
      std::cout << ostr.str();
    }

    return ::cudaFree(ptr);
  }

  cudaError_t _allocate_page(void** ptr, size_t new_size)
  {
    const size_t page_size = 4096;

    // Alloue un multiple de la taille d'une page
    // Comme les transfers de la mémoire unifiée se font par page,
    // cela permet de détecter qu'elle allocation provoque le transfert
    // TODO: vérifier que le début de l'allocation est bien un multiple
    // de la taille de page.
    size_t orig_new_size = new_size;
    size_t n = new_size / page_size;
    if ((new_size % page_size) != 0)
      ++n;
    new_size = (n + 1) * page_size;

    auto r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
    void* p = *ptr;

    const bool do_trace = m_page_allocate_level >= 2;
    if (do_trace) {
      // Utilise un flux spécifique pour être sur que les affichages ne seront pas mélangés
      // en cas de multi-threading
      std::ostringstream ostr;
      ostr << "MALLOC_MANAGED=" << p << " size=" << orig_new_size;
      if (m_page_allocate_level >= 3) {
        String s = platform::getStackTrace();
        ostr << " stack=" << s;
      }
      ostr << "\n";
      std::cout << ostr.str();
    }

    // Indique qu'on privilégie l'allocation sur le GPU mais que le CPU
    // accédera aussi aux données.
    // TODO: regarder pour faire cela tout le temps.
    cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    //cudaMemAdvise(p, new_size, cudaMemAdviseSetReadMostly, 0);
    return r;
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
  cudaError_t _deallocate(void* ptr, MemoryAllocationArgs) override
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

  cudaError_t _allocate(void** ptr, size_t new_size, MemoryAllocationArgs) override
  {
    return ::cudaMalloc(ptr, new_size);
  }
  cudaError_t _deallocate(void* ptr, MemoryAllocationArgs) override
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

void initializeCudaMemoryAllocators()
{
  unified_memory_cuda_memory_allocator.initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
