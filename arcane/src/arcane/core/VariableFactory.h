// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableFactory.h                                            C) 2000-2020 */
/*                                                                           */
/* Fabrique d'une variable d'un type donné.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEFACTORY_H
#define ARCANE_VARIABLEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/IVariableFactory.h"
#include "arcane/core/VariableTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Enregistreur d'une fabrique de variables.
 */
class VariableFactory
: public IVariableFactory
{
 public:

  /*!
   * \brief Crée une fabrique une variable.
   *
   * \param func fonction créant la variable
   * \param item_kind genre d'entité de la variable
   * \param data_type type de donnée de la variable
   * \param dimension dimension de la variable
   * \param multi_tag tag indiquant s'il s'agit d'un tableau à taille multiple.
   * \param is_partial indique s'il s'agit d'une variable partielle.
   */
  ARCCORE_DEPRECATED_2020("Use overload with 'VariableTypeInfo' argument")
  VariableFactory(VariableFactoryFunc func,eDataType data_type,
                  eItemKind item_kind,Integer dimension,Integer multi_tag,bool is_partial);
  VariableFactory(VariableFactoryFunc func,const VariableTypeInfo& var_type_info);

 public:

  VariableRef* createVariable(const VariableBuildInfo& name) override;
  eItemKind itemKind() const override { return m_variable_type_info.itemKind(); }
  eDataType dataType() const override { return m_variable_type_info.dataType(); }
  Integer dimension() const override { return m_variable_type_info.dimension(); }
  Integer multiTag() const override { return m_variable_type_info.multiTag(); }
  bool isPartial() const { return m_variable_type_info.isPartial(); }
  const String& fullTypeName() const override { return m_full_type_name; }
  VariableTypeInfo variableTypeInfo() const override { return m_variable_type_info; }

 private:
  
  //! Fonction de création du IVariableFactoryFactory
  VariableFactoryFunc m_function;

  //! Informations sur le type de la variable
  VariableTypeInfo m_variable_type_info;

  //! Nom complet du type de la variable
  String m_full_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
