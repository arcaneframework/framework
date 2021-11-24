// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
 * \brief Allocateur spécifique pour 'Hip'.
 *
 * Cet allocateur utilise 'hipMallocManaged' au lieu de 'malloc'
 * pour les allocations.
 */
class HipMemoryAllocator
: public Arccore::AlignedMemoryAllocator
{
 public:
  HipMemoryAllocator() : AlignedMemoryAllocator(128){}

  bool hasRealloc() const override { return false; }
  void* allocate(size_t new_size) override
  {
    void* out = nullptr;
    ARCANE_CHECK_HIP(::hipMallocManaged(&out,new_size,hipMemAttachGlobal));
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
    ARCANE_CHECK_HIP(::hipFree(ptr));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HipMemoryAllocator default_hip_memory_allocator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::IMemoryAllocator*
getHipMemoryAllocator()
{
  return &default_hip_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
