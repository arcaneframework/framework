// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableParallelOperation.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a class for parallel operations on variables.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEPARALLELOPERATION_H
#define ARCANE_CORE_IVARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a class for parallel operations on variables.
 *
 * These operations are collective.
 *
 * Before performing the operation, the entity family must be positioned
 * (setItemFamily()), then add the list of variables on which the operations
 * will be performed.
 */
class ARCANE_CORE_EXPORT IVariableParallelOperation
{
 public:

  virtual ~IVariableParallelOperation() = default; //!< Releases resources.

 public:

  virtual void build() =0; //!< Constructs the instance

 public:

  /*!
   * \brief Positions the entity family on which the operation is to be performed.
   * 
   * The family must be positioned before adding variables.
   * It can only be done once.
   */
  virtual void setItemFamily(IItemFamily* family) =0;
  
  //! Entity family on which the operation is performed
  virtual IItemFamily* itemFamily() =0;

  //! Adds a variable to the list of variables concerned by the operation
  virtual void addVariable(IVariable* variable) =0;

  //! Applies the operation.
  virtual void applyOperation(IDataOperation* operation) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
