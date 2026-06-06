// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableFactory.h                                            C) 2000-2025 */
/*                                                                           */
/* Factory for a variable of a given type.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEFACTORY_H
#define ARCANE_CORE_VARIABLEFACTORY_H
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
 * \brief Registerer of a variable factory.
 */
class VariableFactory
: public IVariableFactory
{
 public:

  /*!
   * \brief Creates a variable factory.
   *
   * \param func function creating the variable
   * \param item_kind kind of the variable entity
   * \param data_type data type of the variable
   * \param dimension dimension of the variable
   * \param multi_tag tag indicating if it is a multi-sized array.
   * \param is_partial indicates if it is a partial variable.
   */
  ARCCORE_DEPRECATED_2020("Use overload with 'VariableTypeInfo' argument")
  VariableFactory(VariableFactoryFunc func, eDataType data_type,
                  eItemKind item_kind, Integer dimension, Integer multi_tag, bool is_partial);
  VariableFactory(VariableFactoryFunc func, const VariableTypeInfo& var_type_info);

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

  //! Creation function for IVariableFactoryFactory
  VariableFactoryFunc m_function;

  //! Information about the variable type
  VariableTypeInfo m_variable_type_info;

  //! Full name of the variable type
  String m_full_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
