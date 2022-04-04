// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableTypeInfo.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Infos caractérisant le type d'une variable.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableTypeInfo.h"

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/String.h"

#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/VariableInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableTypeInfo::
_buildFullTypeName() const
{
  StringBuilder full_type_b;
  full_type_b = dataTypeName(dataType());
  full_type_b += ".";
  full_type_b += itemKindName(itemKind());
  full_type_b += ".";
  full_type_b += dimension();
  full_type_b += ".";
  full_type_b += multiTag();
  if (isPartial())
    full_type_b += ".Partial";
  return full_type_b.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableTypeInfo::
fullName() const
{
  return _buildFullTypeName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo VariableTypeInfo::
_internalDefaultDataStorage() const
{
  return VariableInfo::_internalGetStorageTypeInfo(m_data_type,m_dimension,m_multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

