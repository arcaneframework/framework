// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableFactoryRegisterer.cc                                (C) 2000-2020 */
/*                                                                           */
/* Singleton permettant d'enregister une fabrique de variable.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/VariableFactoryRegisterer.h"
#include "arcane/VariableFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableFactoryRegisterer* arcaneFirstVariableFactory = nullptr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableFactoryRegisterer::
VariableFactoryRegisterer(IVariableFactory::VariableFactoryFunc func,
                          const VariableTypeInfo& var_type_info)
: m_function(func)
, m_variable_type_info(var_type_info)
, m_previous(nullptr)
, m_next(nullptr)
{
  if (!arcaneFirstVariableFactory){
    arcaneFirstVariableFactory = this;
  }
  else{
    VariableFactoryRegisterer* next = arcaneFirstVariableFactory->nextVariableFactory();
    setNextVariableFactory(arcaneFirstVariableFactory);
    arcaneFirstVariableFactory = this;
    if (next)
      next->setPreviousVariableFactory(this);
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableFactory* VariableFactoryRegisterer::
createFactory()
{
  return new VariableFactory(m_function,m_variable_type_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableFactoryRegisterer* VariableFactoryRegisterer::
firstVariableFactory()
{
  return arcaneFirstVariableFactory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

