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

#if defined(ARCANE_COMPILING_HIP)
#include "arcane/accelerator/hip/HipAccelerator.h"
#endif
#if defined(ARCANE_COMPILING_CUDA)
#include "arcane/accelerator/cuda/CudaAccelerator.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA)

cudaStream_t CudaUtils::
toNativeStream(RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::CUDA)
    ARCANE_FATAL("RunQueue is not a CUDA queue");
  cudaStream_t* s = reinterpret_cast<cudaStream_t*>(queue->platformStream());
  if (!s)
    ARCANE_FATAL("Null stream");
  return *s;
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_HIP)

hipStream_t HipUtils::
toNativeStream(RunQueue* queue)
{
  eExecutionPolicy p = eExecutionPolicy::None;
  if (queue)
    p = queue->executionPolicy();
  if (p != eExecutionPolicy::HIP)
    ARCANE_FATAL("RunQueue is not a HIP queue");
  hipStream_t* s = reinterpret_cast<hipStream_t*>(queue->platformStream());
  if (!s)
    ARCANE_FATAL("Null stream");
  return *s;
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DeviceStorageBase::
_copyToAsync(Span<std::byte> destination, Span<const std::byte> source, RunQueue* queue)
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

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
