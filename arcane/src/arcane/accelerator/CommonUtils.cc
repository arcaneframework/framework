﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonUtils.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Fonctions/Classes utilitaires communes à tout les runtimes.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/CommonUtils.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/accelerator/core/NativeStream.h"
#include "arcane/accelerator/CommonUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \namespace Arcane::Accelerator::AcceleratorUtils
 *
 * \brief Espace de nom pour les méthodes utilitaires des accélérateurs.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA)

cudaStream_t CudaUtils::
toNativeStream(const NativeStream& v)
{
  cudaStream_t* s = reinterpret_cast<cudaStream_t*>(v.m_native_pointer);
  if (!s)
    ARCANE_FATAL("Null CUDA stream");
  return *s;
}

cudaStream_t CudaUtils::
toNativeStream(const RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::CUDA)
    ARCANE_FATAL("RunQueue is not a CUDA queue");
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

#if defined(ARCANE_COMPILING_HIP)

hipStream_t HipUtils::
toNativeStream(const NativeStream& v)
{
  hipStream_t* s = reinterpret_cast<hipStream_t*>(v.m_native_pointer);
  if (!s)
    ARCANE_FATAL("Null HIP stream");
  return *s;
}

hipStream_t HipUtils::
toNativeStream(const RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::HIP)
    ARCANE_FATAL("RunQueue is not a HIP queue");
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

#if defined(ARCANE_COMPILING_SYCL)

sycl::queue SyclUtils::
toNativeStream(const NativeStream& v)
{
  sycl::queue* s = reinterpret_cast<sycl::queue*>(v.m_native_pointer);
  if (!s)
    ARCANE_FATAL("Null SYCL stream");
  return *s;
}

sycl::queue SyclUtils::
toNativeStream(const RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::SYCL)
    ARCANE_FATAL("RunQueue is not a SYCL queue");
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
#if defined(ARCANE_COMPILING_CUDA)
  cudaStream_t stream = CudaUtils::toNativeStream(queue);
  ARCANE_CHECK_CUDA(::cudaMemcpyAsync(destination.data(), source.data(), source.size(), cudaMemcpyDeviceToHost, stream));
#elif defined(ARCANE_COMPILING_HIP)
  hipStream_t stream = HipUtils::toNativeStream(queue);
  ARCANE_CHECK_HIP(::hipMemcpyAsync(destination.data(), source.data(), source.size(), hipMemcpyDefault, stream));
#else
  ARCANE_UNUSED(destination);
  ARCANE_UNUSED(source);
  ARCANE_UNUSED(queue);
  ARCANE_FATAL("No valid implementation for copy");
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

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
