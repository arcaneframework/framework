// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableTypeInfo.cc                                (C) 2000-2022 */
/*                                                                           */
/* Informations caractérisants le type d'une variable matériaux.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialVariableTypeInfo.h"

#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MaterialVariableTypeInfo::
_buildFullTypeName() const
{
  StringBuilder full_type_b;
  full_type_b = dataTypeName(dataType());
  full_type_b += ".";
  full_type_b += itemKindName(itemKind());
  full_type_b += ".";
  full_type_b += dimension();
  full_type_b += ".";

  //! Variable ayant des valeurs sur les milieux et matériaux
  switch (m_mat_var_space) {
  case MatVarSpace::MaterialAndEnvironment:
    full_type_b += "MatEnv";
    break;
  case MatVarSpace::Environment:
    full_type_b += "Env";
    break;
  default:
    ARCANE_FATAL("Unknown MatVarSpace {0}", (int)m_mat_var_space);
  }

  return full_type_b.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MaterialVariableTypeInfo::
fullName() const
{
  return _buildFullTypeName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
