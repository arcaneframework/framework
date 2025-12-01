// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorUtils.h                                          (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires communes à tous les runtimes.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_ACCELERATORUTILS_H
#define ARCCORE_ACCELERATOR_ACCELERATORUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

#if defined(ARCCORE_COMPILING_HIP)
#include <hip/hip_runtime.h>
#endif
#if defined(ARCCORE_COMPILING_CUDA)
#endif
#if defined(ARCCORE_COMPILING_SYCL)
#include <sycl/sycl.hpp>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)
class ARCCORE_ACCELERATOR_EXPORT CudaUtils
{
 public:

  static cudaStream_t toNativeStream(const RunQueue* queue);
  static cudaStream_t toNativeStream(const RunQueue& queue);
  static cudaStream_t toNativeStream(const NativeStream& v);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_HIP)
class ARCCORE_ACCELERATOR_EXPORT HipUtils
{
 public:

  static hipStream_t toNativeStream(const RunQueue* queue);
  static hipStream_t toNativeStream(const RunQueue& queue);
  static hipStream_t toNativeStream(const NativeStream& v);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)
class ARCCORE_ACCELERATOR_EXPORT SyclUtils
{
 public:

  static sycl::queue toNativeStream(const RunQueue* queue);
  static sycl::queue toNativeStream(const RunQueue& queue);
  static sycl::queue toNativeStream(const NativeStream& v);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::AcceleratorUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)
/*!
 * \brief Retourne l'instance de cudaStream_t associée à \q queue.
 *
 * Une exception est levée si queue.executionPolicy() != eExecutionPolicy::CUDA.
 */
inline cudaStream_t
toCudaNativeStream(const RunQueue& queue)
{
  return Impl::CudaUtils::toNativeStream(queue);
}
#endif

#if defined(ARCCORE_COMPILING_HIP)
/*!
 * \brief Retourne l'instance de hipStream_t associée à \q queue.
 *
 * Une exception est levée si queue.executionPolicy() != eExecutionPolicy::HIP.
 */
inline hipStream_t
toHipNativeStream(const RunQueue& queue)
{
  return Impl::HipUtils::toNativeStream(queue);
}
#endif

#if defined(ARCCORE_COMPILING_SYCL)
/*!
 * \brief Retourne l'instance de hipStream_t associée à \q queue.
 *
 * Une exception est levée si queue.executionPolicy() != eExecutionPolicy::SYCL.
 */
inline sycl::queue
toSyclNativeStream(const RunQueue& queue)
{
  return Impl::SyclUtils::toNativeStream(queue);
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
