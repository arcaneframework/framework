// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
class CudaMemoryAllocator
: public Arccore::AlignedMemoryAllocator
{
 public:
  CudaMemoryAllocator() : AlignedMemoryAllocator(128){}

  bool hasRealloc() const override { return false; }
  void* allocate(size_t new_size) override
  {
    void* out = nullptr;
    ARCANE_CHECK_CUDA(::cudaMallocManaged(&out,new_size,cudaMemAttachGlobal));
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128)!=0)
      ARCANE_FATAL("Bad alignment for CUDA allocator: offset={0}",(a % 128));
    return out;
  }
  void* reallocate(void* current_ptr,size_t new_size) override
  {
    deallocate(current_ptr);
    return allocate(new_size);
  }
  void deallocate(void* ptr) override
  {
    ARCANE_CHECK_CUDA(::cudaFree(ptr));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CudaMemoryAllocator default_cuda_memory_allocator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::IMemoryAllocator*
getCudaMemoryAllocator()
{
  return &default_cuda_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
