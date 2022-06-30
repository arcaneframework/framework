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
#include "arcane/core/materials/IMeshMaterialVariableFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialVariableFactory
: public IMeshMaterialVariableFactory
{
  using CreateFunc = MeshMaterialVariableFactoryVariableRefCreateFunc;

 public:

  MeshMaterialVariableFactory(const MaterialVariableTypeInfo& var_type_info,
                              CreateFunc func)
  : m_variable_type_info(var_type_info)
  , m_function(func)
  {
  }

  IMeshMaterialVariable* createVariable(const MaterialVariableBuildInfo& build_info) override
  {
    ARCANE_CHECK_POINTER(m_function);
    return (*m_function)(build_info);
  }

  MaterialVariableTypeInfo materialVariableTypeInfo() const override
  {
    return m_variable_type_info;
  }

 private:

  MaterialVariableTypeInfo m_variable_type_info;
  CreateFunc m_function;
};

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
  auto* x = new MeshMaterialVariableFactory(m_variable_type_info, m_function);
  return makeRef<IMeshMaterialVariableFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
