// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DotNetRuntimeInitialisationInfoProperties.h                 (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'initialisation du runtime '.Net'.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_DOTNETRUNTIMEINITIALISATIONINFOPROPERTIES_H
#define ARCANE_UTILS_INTERNAL_DOTNETRUNTIMEINITIALISATIONINFOPROPERTIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/DotNetRuntimeInitialisationInfo.h"
#include "arccore/common/internal/PropertyDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DotNetRuntimeInitialisationInfoProperties
: public DotNetRuntimeInitialisationInfo
{
  ARCANE_DECLARE_PROPERTY_CLASS(DotNetRuntimeInitialisationInfo);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
