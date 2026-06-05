// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableFactoryRegisterer.h                     (C) 2000-2022 */
/*                                                                           */
/* Singleton allowing registration of a material variable factory.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLEFACTORYREGISTERER_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLEFACTORYREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/GenericRegisterer.h"

#include "arcane/core/materials/MaterialVariableTypeInfo.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Registerer for a material variable factory..
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableFactoryRegisterer
: public GenericRegisterer<MeshMaterialVariableFactoryRegisterer>
{
 public:

  using BaseInfo = GenericRegisterer<MeshMaterialVariableFactoryRegisterer>::Info;
  static BaseInfo& registererInfo();

  using MeshMaterialVariableFactoryFunc = MeshMaterialVariableFactoryVariableRefCreateFunc;

 public:

 //! Creates a registerer for a MeshMaterialVariableFactory for the type \a var_type_info and for the creation function \a func
  MeshMaterialVariableFactoryRegisterer(MeshMaterialVariableFactoryFunc func,
                                        const MaterialVariableTypeInfo& var_type_info)
  : m_function(func)
  , m_variable_type_info(var_type_info)
  {}

 public:

  //! Creates a factory for this variable type.
  Ref<IMeshMaterialVariableFactory> createFactory();

  //! Information about the variable type
  const MaterialVariableTypeInfo& variableTypeInfo() const { return m_variable_type_info; }

 private:

  //! Creation function of the IMeshMaterialVariableFactoryFactory
  MeshMaterialVariableFactoryFunc m_function;

  //! Information about the variable type
  MaterialVariableTypeInfo m_variable_type_info;

 private:

  static BaseInfo m_global_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
