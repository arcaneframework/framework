// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonUtils.cc                                              (C) 2000-2026 */
/*                                                                           */
/* Fonctions/Classes utilitaires communes à tout les runtimes.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/CommonUtils.h"

#include "arccore/base/FatalErrorException.h"

#include "arccore/common/MemoryUtils.h"
#include "arccore/common/accelerator/NativeStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \namespace Arcane::Accelerator::AcceleratorUtils
 *
 * \brief Espace de nom pour les méthodes utilitaires des accélérateurs.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)

cudaStream_t CudaUtils::
toNativeStream(const NativeStream& v)
{
  cudaStream_t* s = reinterpret_cast<cudaStream_t*>(v.m_native_pointer);
  if (!s)
    ARCCORE_FATAL("Null CUDA stream");
  return *s;
}

cudaStream_t CudaUtils::
toNativeStream(const RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::CUDA)
    ARCCORE_FATAL("RunQueue is not a CUDA queue");
  return toNativeStream(queue->_internalNativeStream());
}

cudaStream_t CudaUtils::
toNativeStream(const RunQueue& queue)
{
  return toNativeStream(&queue);
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_HIP)

hipStream_t HipUtils::
toNativeStream(const NativeStream& v)
{
  hipStream_t* s = reinterpret_cast<hipStream_t*>(v.m_native_pointer);
  if (!s)
    ARCCORE_FATAL("Null HIP stream");
  return *s;
}

hipStream_t HipUtils::
toNativeStream(const RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::HIP)
    ARCCORE_FATAL("RunQueue is not a HIP queue");
  return toNativeStream(queue->_internalNativeStream());
}

hipStream_t HipUtils::
toNativeStream(const RunQueue& queue)
{
  return toNativeStream(&queue);
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

sycl::queue SyclUtils::
toNativeStream(const NativeStream& v)
{
  sycl::queue* s = reinterpret_cast<sycl::queue*>(v.m_native_pointer);
  if (!s)
    ARCCORE_FATAL("Null SYCL stream");
  return *s;
}

sycl::queue SyclUtils::
toNativeStream(const RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::SYCL)
    ARCCORE_FATAL("RunQueue is not a SYCL queue");
  return toNativeStream(queue->_internalNativeStream());
}

sycl::queue SyclUtils::
toNativeStream(const RunQueue& queue)
{
  return toNativeStream(&queue);
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DeviceStorageBase::
_copyToAsync(Span<std::byte> destination, Span<const std::byte> source, const RunQueue& queue)
{
#if defined(ARCCORE_COMPILING_CUDA)
  cudaStream_t stream = Impl::CudaUtils::toNativeStream(queue);
  ARCANE_CHECK_CUDA(::cudaMemcpyAsync(destination.data(), source.data(), source.size(), cudaMemcpyDeviceToHost, stream));
#elif defined(ARCCORE_COMPILING_HIP)
  hipStream_t stream = Impl::HipUtils::toNativeStream(queue);
  ARCANE_CHECK_HIP(::hipMemcpyAsync(destination.data(), source.data(), source.size(), hipMemcpyDefault, stream));
#else
  ARCCORE_UNUSED(destination);
  ARCCORE_UNUSED(source);
  ARCCORE_UNUSED(queue);
  ARCCORE_FATAL("No valid implementation for copy");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GenericDeviceStorage::
GenericDeviceStorage()
: m_storage(MemoryUtils::getDeviceOrHostAllocator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
