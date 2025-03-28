// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.h                                           (C) 2000-2025 */
/*                                                                           */
/* Backend 'HIP' pour les accélérateurs.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_HIP_HIPACCELERATOR_H
#define ARCANE_HIP_HIPACCELERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"

#include <iostream>

#include <hip/hip_runtime.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_hip
#define ARCANE_HIP_EXPORT ARCANE_EXPORT
#else
#define ARCANE_HIP_EXPORT ARCANE_IMPORT
#endif

namespace Arcane::Accelerator::Hip
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_HIP_EXPORT void
arcaneCheckHipErrors(const TraceInfo& ti,hipError_t e);

extern "C++" ARCANE_HIP_EXPORT void
arcaneCheckHipErrorsNoThrow(const TraceInfo& ti,hipError_t e);

#define ARCANE_CHECK_HIP(result) \
  Arcane::Accelerator::Hip::arcaneCheckHipErrors(A_FUNCINFO,result)

//! Verifie \a result et affiche un message d'erreur en cas d'erreur.
#define ARCANE_CHECK_HIP_NOTHROW(result) \
  Arcane::Accelerator::Hip::arcaneCheckHipErrorsNoThrow(A_FUNCINFO,result)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_HIP_EXPORT Arccore::IMemoryAllocator*
getHipMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT Arccore::IMemoryAllocator*
getHipDeviceMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT Arccore::IMemoryAllocator*
getHipUnifiedMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT Arccore::IMemoryAllocator*
getHipHostPinnedMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
