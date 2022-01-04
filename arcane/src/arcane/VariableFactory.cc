// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableFactory.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Fabrique d'une variable d'un type donné.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/VariableFactory.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableFactory::
VariableFactory(VariableFactoryFunc func,eDataType data_type,
                eItemKind item_kind,Integer dimension,Integer multi_tag,
                bool is_partial)
: VariableFactory(func,VariableTypeInfo(item_kind,data_type,dimension,multi_tag,is_partial))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableFactory::
VariableFactory(VariableFactoryFunc func,const VariableTypeInfo& var_type_info)
: m_function(func)
, m_variable_type_info(var_type_info)
{
  m_full_type_name = m_variable_type_info.fullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef* VariableFactory::
createVariable(const VariableBuildInfo& vbi)
{
  VariableRef* var = (*m_function)(vbi);
  return var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
