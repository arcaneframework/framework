// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorUtils.h                                          (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires communes à tous les runtimes.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_ACCELERATORUTILS_H
#define ARCANE_ACCELERATOR_ACCELERATORUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

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
#if defined(ARCANE_COMPILING_SYCL)
#include "arcane/accelerator/sycl/SyclAccelerator.h"
#include <sycl/sycl.hpp>
#if defined(__INTEL_LLVM_COMPILER)
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA)
class ARCANE_ACCELERATOR_EXPORT CudaUtils
{
 public:

  static cudaStream_t toNativeStream(const RunQueue* queue);
  static cudaStream_t toNativeStream(const RunQueue& queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_HIP)
class ARCANE_ACCELERATOR_EXPORT HipUtils
{
 public:

  static hipStream_t toNativeStream(const RunQueue* queue);
  static hipStream_t toNativeStream(const RunQueue& queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)
class ARCANE_ACCELERATOR_EXPORT SyclUtils
{
 public:

  static sycl::queue toNativeStream(const RunQueue* queue);
  static sycl::queue toNativeStream(const RunQueue& queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::AcceleratorUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA)
/*!
 * \brief Retourne l'instance de cudaStream_t associée à \q queue.
 *
 * Une exception est levée si queue.executionPolicy() != eExecutionPolicy::CUDA.
 */
inline cudaStream_t
toCudaNativeStream(const RunQueue& queue)
{
  return impl::CudaUtils::toNativeStream(queue);
}
#endif

#if defined(ARCANE_COMPILING_HIP)
/*!
 * \brief Retourne l'instance de hipStream_t associée à \q queue.
 *
 * Une exception est levée si queue.executionPolicy() != eExecutionPolicy::HIP.
 */
inline hipStream_t
toHipNativeStream(const RunQueue& queue)
{
  return impl::HipUtils::toNativeStream(queue);
}
#endif

#if defined(ARCANE_COMPILING_SYCL)
/*!
 * \brief Retourne l'instance de hipStream_t associée à \q queue.
 *
 * Une exception est levée si queue.executionPolicy() != eExecutionPolicy::SYCL.
 */
inline sycl::queue
toSyclNativeStream(const RunQueue& queue)
{
  return impl::SyclUtils::toNativeStream(queue);
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::AcceleratorUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
