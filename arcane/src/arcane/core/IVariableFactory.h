// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableFactory.h                                           C) 2000-2025 */
/*                                                                           */
/* Interface for a factory of a variable of a given type.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEFACTORY_H
#define ARCANE_CORE_IVARIABLEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a variable factory.
 *
 The instance allows creating a variable based on its data type (dataType()),
 its entity kind (itemKind()), its dimension (dimension()), and its tag if it
 is a multi-array (multiTag()).

 The fullTypeName() operation contains the complete type name, obtained in the
 following way: dataType().itemKind().dimension().multiTag. For example, for a
 real scalar variable on the meshes, the complete type is: \a "Real.Cell.0.0".
 */
class IVariableFactory
{
 public:

  //! Type of the function that creates the variable
  using VariableFactoryFunc = VariableFactoryVariableRefCreateFunc;

 public:

  virtual ~IVariableFactory() = default;

 public:

  //! Creates a variable with the \a build_info and returns its reference.
  virtual VariableRef* createVariable(const VariableBuildInfo& build_info) = 0;

 public:

  //! Kind of the data variables created by this factory
  virtual eItemKind itemKind() const = 0;

  //! Data type of the variable created by this factory
  virtual eDataType dataType() const = 0;

  //! Dimension of the variable created by this factory
  virtual Integer dimension() const = 0;

  //! Multi tag.
  virtual Integer multiTag() const = 0;

  //! Full name of the variable type
  virtual const String& fullTypeName() const = 0;

 public:

  //! Information about the variable type
  virtual VariableTypeInfo variableTypeInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
