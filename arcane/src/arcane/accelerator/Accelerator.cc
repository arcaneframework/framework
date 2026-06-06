// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Accelerator.cc                                              (C) 2000-2025 */
/*                                                                           */
/* General declarations for accelerator support.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/accelerator/Reduce.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/accelerator/SpanViews.h"

#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file RunCommandEnumerate.h
 *
 * \brief Types and macros to manage enumerations of entities on accelerators
 */

/*!
 * \file RunCommandMaterialEnumerate.h
 *
 * \brief Types and macros to manage enumerations of materials and
 * media on accelerators
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::Accelerator::
initializeRunner(Runner& runner, ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  Impl::arccoreInitializeRunner(runner, tm, acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
