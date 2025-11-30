// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.h                                           (C) 2000-2025 */
/*                                                                           */
/* Backend 'CUDA' pour les accélérateurs.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CUDA_CUDAACCELERATOR_H
#define ARCANE_CUDA_CUDAACCELERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"
#include "arccore/base/BaseTypes.h"

#include <cuda_runtime.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_cuda
#define ARCANE_CUDA_EXPORT ARCCORE_EXPORT
#else
#define ARCANE_CUDA_EXPORT ARCCORE_IMPORT
#endif

namespace Arcane::Accelerator::Cuda
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CUDA_EXPORT void
arcaneCheckCudaErrors(const TraceInfo& ti,cudaError_t e);

extern "C++" ARCANE_CUDA_EXPORT void
arcaneCheckCudaErrorsNoThrow(const TraceInfo& ti,cudaError_t e);

//! Vérifie \a result et lance une exception en cas d'erreur
#define ARCANE_CHECK_CUDA(result) \
  Arcane::Accelerator::Cuda::arcaneCheckCudaErrors(A_FUNCINFO,result)

//! Verifie \a result et affiche un message d'erreur en cas d'erreur.
#define ARCANE_CHECK_CUDA_NOTHROW(result) \
  Arcane::Accelerator::Cuda::arcaneCheckCudaErrorsNoThrow(A_FUNCINFO,result)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
#define ARCANE_USING_CUDA13_OR_GREATER
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
