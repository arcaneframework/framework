// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAccelerator.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Backend 'HIP' pour les accélérateurs.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/hip/HipAccelerator.h"

#include "arccore/base/FatalErrorException.h"

#include <iostream>

namespace Arcane::Accelerator::Hip
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
arcaneCheckHipErrors(const TraceInfo& ti,hipError_t e)
{
  if (e!=hipSuccess){
    ARCCORE_FATAL("HIP Error trace={0} e={1} str={2}",ti,e,hipGetErrorString(e));
  }
}

void
arcaneCheckHipErrorsNoThrow(const TraceInfo& ti,hipError_t e)
{
  if (e==hipSuccess)
    return;
  String str = String::format("HIP Error trace={0} e={1} str={2}",ti,e,hipGetErrorString(e));
  FatalErrorException ex(ti,str);
  ex.explain(std::cerr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Hip

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
