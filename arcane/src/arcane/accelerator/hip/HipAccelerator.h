// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.h                                           (C) 2000-2021 */
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

using namespace Arccore;

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

#define ARCANE_CHECK_HIP(result) \
  Arcane::Accelerator::Hip::arcaneCheckHipErrors(A_FUNCINFO,result)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_HIP_EXPORT Arccore::IMemoryAllocator*
getHipMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
