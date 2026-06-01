// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelLoopOptionsProperties.h                             (C) 2000-2025 */
/*                                                                           */
/* Configuration options for parallel loops in multi-threading.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_PARALLELLOOPOPTIONSPROPERTIES_H
#define ARCANE_UTILS_INTERNAL_PARALLELLOOPOPTIONSPROPERTIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParallelLoopOptions.h"
#include "arccore/common/internal/PropertyDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to set the values of ParallelLoopOptions via properties.
 */
class ARCANE_UTILS_EXPORT ParallelLoopOptionsProperties
: public ParallelLoopOptions
{
  ARCANE_DECLARE_PROPERTY_CLASS(ParallelLoopOptions);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
