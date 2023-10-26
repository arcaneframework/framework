// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonUtils.h                                               (C) 2000-2023 */
/*                                                                           */
/* Fonctions/Classes utilitaires communes à tout les runtimes.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_COMMONUTILS_H
#define ARCANE_ACCELERATOR_COMMONUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"

#if defined(ARCANE_COMPILING_HIP)
#include "arcane/accelerator/hip/HipAccelerator.h"
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#endif
#if defined(ARCANE_COMPILING_CUDA)
#include "arcane/accelerator/cuda/CudaAccelerator.h"
#include <cub/cub.cuh>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

#if defined(ARCANE_COMPILING_CUDA)
class ARCANE_ACCELERATOR_EXPORT CudaUtils
{
 public:

  static cudaStream_t toNativeStream(RunQueue* queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_HIP)
class ARCANE_ACCELERATOR_EXPORT HipUtils
{
 public:

  static hipStream_t toNativeStream(RunQueue* queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceStorage
{
 public:

  ~DeviceStorage() ARCANE_NOEXCEPT
  {
    deallocate();
  }

 public:

  void* address() { return m_ptr; }

  void allocate(size_t new_size)
  {
    if (new_size<m_size)
      return;
    deallocate();
#if defined(ARCANE_COMPILING_CUDA)
    ARCANE_CHECK_CUDA(::cudaMalloc(&m_ptr, new_size));
#endif
#if defined(ARCANE_COMPILING_HIP)
    ARCANE_CHECK_HIP(::hipMalloc(&m_ptr, new_size));
#endif
    m_size = new_size;
  }

  void deallocate()
  {
    if (!m_ptr)
      return;
#if defined(ARCANE_COMPILING_CUDA)
    ARCANE_CHECK_CUDA(::cudaFree(m_ptr));
#endif
#if defined(ARCANE_COMPILING_HIP)
    ARCANE_CHECK_HIP(::hipFree(m_ptr));
#endif
    m_ptr = nullptr;
    m_size = 0;
  }

 private:

  void* m_ptr = nullptr;
  size_t m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
