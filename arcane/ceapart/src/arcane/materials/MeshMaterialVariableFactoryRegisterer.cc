// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialFactoryRegisterer.cc                            (C) 2000-2022 */
/*                                                                           */
/* Singleton permettant d'enregister une fabrique de variable matériaux.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MeshMaterialVariableFactoryRegisterer.h"

#include "arcane/utils/Ref.h"
//#include "arcane/materials/MeshMaterialVariableFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typename MeshMaterialVariableFactoryRegisterer::BaseInfo MeshMaterialVariableFactoryRegisterer::m_global_infos;

typename MeshMaterialVariableFactoryRegisterer::BaseInfo& MeshMaterialVariableFactoryRegisterer::
registererInfo()
{
  return m_global_infos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMeshMaterialVariableFactory> MeshMaterialVariableFactoryRegisterer::
createFactory()
{
  return {};
  //return new MeshMaterialFactory(m_function,m_variable_type_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
