// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationInfoProperties.h                                 (C) 2000-2025 */
/*                                                                           */
/* Informations sur une application.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_APPLICATIONINFOPROPERTIES_H
#define ARCANE_UTILS_INTERNAL_APPLICATIONINFOPROPERTIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arccore/common/internal/PropertyDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une application.
 */
class ARCANE_UTILS_EXPORT ApplicationInfoProperties
: public ApplicationInfo
{
  ARCANE_DECLARE_PROPERTY_CLASS(ApplicationInfo);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
