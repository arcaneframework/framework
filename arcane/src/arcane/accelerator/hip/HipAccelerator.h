// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAccelerator.h                                            (C) 2000-2025 */
/*                                                                           */
/* Backend 'ROCM/HIP' pour les accélérateurs.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_HIP_HIPACCELERATOR_H
#define ARCANE_HIP_HIPACCELERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"
#include "arccore/base/BaseTypes.h"

#include <hip/hip_runtime.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_hip
#define ARCANE_HIP_EXPORT ARCCORE_EXPORT
#else
#define ARCANE_HIP_EXPORT ARCCORE_IMPORT
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

extern "C++" ARCANE_HIP_EXPORT IMemoryAllocator*
getHipMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT IMemoryAllocator*
getHipDeviceMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT IMemoryAllocator*
getHipUnifiedMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT IMemoryAllocator*
getHipHostPinnedMemoryAllocator();

extern "C++" ARCANE_HIP_EXPORT void
initializeHipMemoryAllocators();

extern "C++" ARCANE_HIP_EXPORT void
finalizeHipMemoryAllocators(ITraceMng* tm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
