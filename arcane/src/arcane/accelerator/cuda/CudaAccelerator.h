// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.h                                           (C) 2000-2020 */
/*                                                                           */
/* Backend 'CUDA' pour les accélérateurs.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CUDA_CUDAACCELERATOR_H
#define ARCANE_CUDA_CUDAACCELERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"

#include <iostream>

#include <cuda_runtime.h>

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_cuda
#define ARCANE_CUDA_EXPORT ARCANE_EXPORT
#else
#define ARCANE_CUDA_EXPORT ARCANE_IMPORT
#endif

namespace Arcane::Accelerator::Cuda
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CUDA_EXPORT void
arcaneCheckCudaErrors(const TraceInfo& ti,cudaError_t e);

#define ARCANE_CHECK_CUDA(result) \
  Arcane::Accelerator::Cuda::arcaneCheckCudaErrors(A_FUNCINFO,result)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CUDA_EXPORT Arccore::IMemoryAllocator*
getCudaMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
