// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableFactory.h                              (C) 2000-2025 */
/*                                                                           */
/* Interface of a material variable factory.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLEFACTORY_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a material variable factory.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableFactory
{
 public:

  virtual ~IMeshMaterialVariableFactory() = default;

 public:

  //! Create a material variable
  virtual IMeshMaterialVariable*
  createVariable(const MaterialVariableBuildInfo& build_info) = 0;

  //! Information about the created variable type
  virtual MaterialVariableTypeInfo materialVariableTypeInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
