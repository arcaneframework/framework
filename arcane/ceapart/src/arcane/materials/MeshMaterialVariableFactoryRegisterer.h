// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableFactoryRegisterer.h                     (C) 2000-2022 */
/*                                                                           */
/* Singleton permettant d'enregister une fabrique de variable matériaux.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLEFACTORYREGISTERER_H
#define ARCANE_MATERiALS_MESHMATERIALVARIABLEFACTORYREGISTERER_H
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
 * \brief Enregistreur d'une fabrique de variables matériaux..
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableFactoryRegisterer
: public GenericRegisterer<MeshMaterialVariableFactoryRegisterer>
{
 public:

  using BaseInfo = GenericRegisterer<MeshMaterialVariableFactoryRegisterer>::Info;
  static BaseInfo& registererInfo();

  using MeshMaterialVariableFactoryFunc = MeshMaterialVariableFactoryVariableRefCreateFunc;

 public:

 //! Crée un enregistreur pour une MeshMaterialVariableFactory pour le type \a var_type_info et pour fonction de création \a func
  MeshMaterialVariableFactoryRegisterer(MeshMaterialVariableFactoryFunc func,
                                        const MaterialVariableTypeInfo& var_type_info)
  : m_function(func)
  , m_variable_type_info(var_type_info)
  {}

 public:

  //! Créé une fabrique pour ce type de variable.
  Ref<IMeshMaterialVariableFactory> createFactory();

  //! Informations sur le type de la variable
  const MaterialVariableTypeInfo& variableTypeInfo() const { return m_variable_type_info; }

 private:

  //! Fonction de création du IMeshMaterialVariableFactoryFactory
  MeshMaterialVariableFactoryFunc m_function;

  //! Informations sur le type de la variable
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

