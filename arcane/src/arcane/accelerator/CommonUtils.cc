// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonUtils.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Fonctions/Classes utilitaires communes à tout les runtimes.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/CommonUtils.h"

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

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
