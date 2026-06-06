// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExpressionImpl.h                                           (C) 2000-2014 */
/*                                                                           */
/* Interface for the different implementations of an expression.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_IEXPRESSIONIMPL_H
#define ARCANE_EXPR_IEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ExpressionResult;
class Expression;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for the different implementations of an expression.
 */
class ARCANE_EXPR_EXPORT IExpressionImpl
{
 protected:

  //! Releases resources. Only called by a removeRef()
  virtual ~IExpressionImpl() {}

 public:

  virtual void assign(IExpressionImpl* expr) = 0;
  virtual void assign(IExpressionImpl* expr, IntegerConstArrayView indices) = 0;
  /*! \brief Number of elements in the vector
   *
   * If the expression is a vector and a terminal symbol (a leaf),
   * it returns its number of elements. Otherwise, it returns 0.
   */
  virtual Integer vectorSize() const = 0;

  virtual void dumpIf(IExpressionImpl* test_expr, Array<Expression>& exprs) = 0;
  virtual void apply(ExpressionResult* result) = 0;
  virtual void addRef() = 0;
  virtual void removeRef() = 0;
  virtual void setTrace(bool v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
