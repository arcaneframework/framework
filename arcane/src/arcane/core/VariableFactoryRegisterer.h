// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableFactoryRegisterer.h                                 (C) 2000-2025 */
/*                                                                           */
/* Singleton allowing registration of a variable factory.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEFACTORYREGISTERER_H
#define ARCANE_CORE_VARIABLEFACTORYREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Variable factory registrar.
 *
Each instance of this type must be a global variable that references
a variable creation function of a given type. Since each instance
is global, its creation happens before entering main(). Therefore, no
operations using external objects or even dynamic allocations should be performed
because the already created objects are unknown.

The class is designed so that all its instances are linked by a
linked list. The first link is retrievable via firstVariableFactory();
*/
class ARCANE_CORE_EXPORT VariableFactoryRegisterer
{
 public:

  using VariableFactoryFunc = VariableFactoryVariableRefCreateFunc;

 public:

  //! Creates a registrar for a VariableFactory for the type \a var_type_info
  //! and for the creation function \a func
  VariableFactoryRegisterer(VariableFactoryFunc func, const VariableTypeInfo& var_type_info);

 public:

  /*!
   * \brief Creates a factory for this variable type.
   *
   * The factory must be destroyed by the delete operator when it is
   * no longer used.
   */
  IVariableFactory* createFactory();

  //! Previous VariableFactory (0 if the first)
  VariableFactoryRegisterer* previousVariableFactory() const { return m_previous; }

  //! Next VariableFactory (0 if the last)
  VariableFactoryRegisterer* nextVariableFactory() const { return m_next; }

  //! Kind of data variables of the variable created by this factory
  eItemKind itemKind() const { return m_variable_type_info.itemKind(); }

  //! Data type of the variable created by this factory
  eDataType dataType() const { return m_variable_type_info.dataType(); }

  //! Dimension of the variable created by this factory
  Integer dimension() const { return m_variable_type_info.dimension(); }

  //! Tag indicating the multiple type (0 if not multiple, 1 if multiple, 2 if multiple deprecated)
  Integer multiTag() const { return m_variable_type_info.multiTag(); }

  //! Indicates if the factory is for a partial variable.
  bool isPartial() const { return m_variable_type_info.isPartial(); }

  //! Information about the variable type
  const VariableTypeInfo& variableTypeInfo() const { return m_variable_type_info; }

  /*!
   * \brief Positions the previous VariableFactory.
   *
   * This method is automatically called by IVariableFactoryRegistry.
   */
  void setPreviousVariableFactory(VariableFactoryRegisterer* s) { m_previous = s; }

  /*!
   *  \brief Positions the next VariableFactory
   *
   * This method is automatically called by IVariableFactoryRegistry.
   */
  void setNextVariableFactory(VariableFactoryRegisterer* s) { m_next = s; }

 public:

  static VariableFactoryRegisterer* firstVariableFactory();

 private:

  //! Creation function for IVariableFactoryFactory
  VariableFactoryFunc m_function;

  //! Information about the variable type
  VariableTypeInfo m_variable_type_info;

  //! Previous VariableFactory
  VariableFactoryRegisterer* m_previous;

  //! Next VariableFactory
  VariableFactoryRegisterer* m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
