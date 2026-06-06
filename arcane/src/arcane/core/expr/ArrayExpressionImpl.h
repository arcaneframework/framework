// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExpressionImpl.h                                       (C) 2000-2005 */
/*                                                                           */
/* Expression handling an array.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_ARRAYEXPRESSIONIMPL_H
#define ARCANE_EXPR_ARRAYEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/Expression.h"
#include "arcane/expr/ExpressionResult.h"
#include "arcane/expr/BadOperandException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArrayVariant;
class ArrayOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArrayExpressionImpl
: public ExpressionImpl
{
 public:

  ArrayExpressionImpl(ArrayVariant* var);
  ~ArrayExpressionImpl();

 public:

  virtual void assign(IExpressionImpl* expr);
  virtual void assign(IExpressionImpl*, ConstArrayView<Integer> indices);
  virtual void apply(ExpressionResult* result);
  virtual Integer vectorSize() const;

 private:

  ArrayVariant* m_variant;
  ArrayOperator* m_op;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
