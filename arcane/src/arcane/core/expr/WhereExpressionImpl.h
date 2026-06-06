// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WhereExpressionImpl.h                                       (C) 2000-2004 */
/*                                                                           */
/* Implementation of a conditional expression.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_WHEREEXPRESSIONIMPL_H
#define ARCANE_EXPR_WHEREEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/expr/BadOperandException.h"
#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/ExpressionResult.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class WhereOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of a binary expression
 */
class ARCANE_EXPR_EXPORT WhereExpressionImpl
: public ExpressionImpl
{
 public:

  WhereExpressionImpl(IExpressionImpl* test,
                      IExpressionImpl* iftrue,
                      IExpressionImpl* iffalse);

 public:

  virtual void assign(IExpressionImpl*) {}
  virtual void assign(IExpressionImpl*, IntegerConstArrayView) {}
  virtual void apply(ExpressionResult* result);
  virtual Integer vectorSize() const { return 0; }

 private:

  Expression m_test; //!< Test expression
  Expression m_iftrue; //!< Expression evaluated when the test is positive
  Expression m_iffalse; //!< Expression evaluated when the test is negative
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic operator for conditional expressions.
 */
class WhereOperator
{
 public:

  virtual ~WhereOperator() {}
  virtual void evaluate(ExpressionResult* res,
                        ArrayVariant* test,
                        ArrayVariant* iftrue,
                        ArrayVariant* iffalse) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
class WhereOperatorT
: public WhereOperator
{
 public:

  virtual ~WhereOperatorT() {}
  virtual void evaluate(ExpressionResult* res,
                        ArrayVariant* test,
                        ArrayVariant* iftrue,
                        ArrayVariant* iffalse)
  {
    // verification of operation validity
    if (test->type() != ArrayVariant::TBool)
      throw BadOperandException("WhereOperatorT::evaluate");

    Integer size = res->size();
    if (size != test->size())
      throw BadOperandException("WhereOperatorT::evaluate");

    if (iftrue->type() || iffalse->type())
      throw BadOperandException("WhereOperatorT::evaluate");

    // allocation of the result based on the type of the if
    res->allocate(iftrue->type());

    // retrieval of operand values
    ArrayView<bool> test_val;
    test->value(test_val);
    ArrayView<T> res_val;
    res->data()->value(res_val);
    ArrayView<T> iftrue_val;
    iftrue->value(iftrue_val);
    ArrayView<T> iffalse_val;
    iffalse->value(iffalse_val);

    Integer false_i = 0;
    Integer true_i = 0;
    for (Integer i = 0; i < size; ++i)
      test_val[i] ? res_val[i] = iftrue_val[true_i++]
                  : res_val[i] = iffalse_val[false_i++];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
