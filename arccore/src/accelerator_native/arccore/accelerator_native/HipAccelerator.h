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
#ifndef ARCCORE_HIP_HIPACCELERATOR_H
#define ARCCORE_HIP_HIPACCELERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"
#include "arccore/base/BaseTypes.h"

#include <hip/hip_runtime.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_COMPONENT_arcane_hip
#define ARCCORE_HIP_EXPORT ARCCORE_EXPORT
#else
#define ARCCORE_HIP_EXPORT ARCCORE_IMPORT
#endif

namespace Arcane::Accelerator::Hip
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_HIP_EXPORT void
arcaneCheckHipErrors(const TraceInfo& ti,hipError_t e);

extern "C++" ARCCORE_HIP_EXPORT void
arcaneCheckHipErrorsNoThrow(const TraceInfo& ti,hipError_t e);

//! Vérifie \a result et lance une exception en cas d'erreur
#define ARCCORE_CHECK_HIP(result) \
  Arcane::Accelerator::Hip::arcaneCheckHipErrors(A_FUNCINFO,result)

//! Verifie \a result et affiche un message d'erreur en cas d'erreur.
#define ARCCORE_CHECK_HIP_NOTHROW(result) \
  Arcane::Accelerator::Hip::arcaneCheckHipErrorsNoThrow(A_FUNCINFO,result)

#define ARCANE_CHECK_HIP(result) ARCCORE_CHECK_HIP((result))
#define ARCANE_CHECK_HIP_NOTHROW(result) ARCCORE_CHECK_HIP_NOTHROW((result))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
