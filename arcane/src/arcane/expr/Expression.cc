// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Expression.cc                                               (C) 2000-2014 */
/*                                                                           */
/* Référence à une expression.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"

#include "arcane/expr/Expression.h"
#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/ExpressionResult.h"
#include "arcane/expr/BinaryExpressionImpl.h"
#include "arcane/expr/UnaryExpressionImpl.h"
#include "arcane/expr/LitteralExpressionImpl.h"
#include "arcane/expr/WhereExpressionImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression::
Expression()
: m_expression(0)
{
}

Expression::
Expression(Real v)
{
  m_expression = new LitteralExpressionImpl(v);
  m_expression->addRef();
}

Expression::
Expression(IExpressionImpl* expr)
: m_expression(expr)
{
  if (m_expression)
    m_expression->addRef();
}

Expression::
Expression(const Expression& expr)
: m_expression(expr.m_expression)
{
  if (m_expression)
    m_expression->addRef();
}

void Expression::
operator=(const Expression& expr)
{
  IExpressionImpl* nex = expr.m_expression;
  if (nex)
    nex->addRef();
  if (m_expression)
    m_expression->removeRef();
  m_expression = nex;
}

Expression::
~Expression()
{
  if (m_expression)
    m_expression->removeRef();
}

void Expression::
assign(const Expression& expr)
{
  m_expression->assign(expr.m_expression);
}

void Expression::
assign(const Expression& expr,const Array<Integer>& indices)
{
  m_expression->assign(expr.m_expression, indices);
}

void Expression::
assign(Real val)
{
  Expression expr(literal(val));
  m_expression->assign(expr.m_expression);
}

void Expression::
apply(ExpressionResult* result)
{
  m_expression->apply(result);
}

void Expression::
dumpIf(const Expression& test_expr,Array<Expression>& exprs)
{
  m_expression->dumpIf(test_expr.m_expression,exprs);
}

void Expression::
dumpIf(const Expression& test_expr)
{
  UniqueArray<Expression> exprs;
  m_expression->dumpIf(test_expr.m_expression,exprs);
}

IExpressionImpl* Expression::
operator->() const { return m_expression;}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression Expression::
operator-()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::UnarySubstract);
}

Expression Expression::
inverse()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Inverse);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression Expression::
acos()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Acos);
}

Expression Expression::
asin()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Asin);
}

Expression Expression::
atan()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Atan);
}

Expression Expression::
ceil()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Ceil);
}

Expression Expression::
cos()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Cos);
}

Expression Expression::
cosh()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Cosh);
}

Expression Expression::
exp()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Exp);
}

Expression Expression::
fabs()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Fabs);
}

Expression Expression::
floor()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Floor);
}

Expression Expression::
log()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Log);
}

Expression Expression::
log10()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Log10);
}

Expression Expression::
sin()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Sin);
}

Expression Expression::
sinh()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Sinh);
}

Expression Expression::
sqrt()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Sqrt);
}

Expression Expression::
tan()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Tan);
}

Expression Expression::
tanh()
{
  return new UnaryExpressionImpl(m_expression,
                                 UnaryExpressionImpl::Tanh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression Expression::
operator+(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,ex1. m_expression,
                                  BinaryExpressionImpl::Add);
}

Expression Expression::
operator-(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,ex1. m_expression,
                                  BinaryExpressionImpl::Substract);
}

Expression Expression::
operator*(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,ex1. m_expression,
                                  BinaryExpressionImpl::Multiply);
}

Expression Expression::
operator/(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,ex1. m_expression,
                                  BinaryExpressionImpl::Divide);
}

Expression Expression::
eq(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,ex1. m_expression,
                                  BinaryExpressionImpl::Equal);
}

Expression Expression::
lt(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,ex1. m_expression,
                                  BinaryExpressionImpl::LessThan);
}

Expression Expression::
gt(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression, ex1.m_expression,
                                  BinaryExpressionImpl::GreaterThan);
}

Expression Expression::
lte(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression, ex1.m_expression,
                                  BinaryExpressionImpl::LessOrEqualThan);
}

Expression Expression::
gte(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression, ex1.m_expression,
                                  BinaryExpressionImpl::GreaterOrEqualThan);
}
Expression Expression::
eand(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression, ex1.m_expression,
                                  BinaryExpressionImpl::And);
}
Expression Expression::
eor(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression, ex1.m_expression,
                                  BinaryExpressionImpl::Or);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression Expression::
minimum(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,
                                  ex1.m_expression,
                                  BinaryExpressionImpl::Minimum);
}

Expression Expression::
maximum(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,
                                  ex1.m_expression,
                                  BinaryExpressionImpl::Maximum);
}

Expression Expression::
pow(Expression ex1)
{
  return new BinaryExpressionImpl(m_expression,
                                  ex1.m_expression,
                                  BinaryExpressionImpl::Pow);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression Expression::
ifelse(Expression ex1,Expression ex2)
{
  return new WhereExpressionImpl(m_expression,
                                 ex1.m_expression,
                                 ex2.m_expression);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Expression Expression::
literal(Real v)
{
  return new LitteralExpressionImpl(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Expression::
setTrace(bool v)
{
  m_expression->setTrace(v);
}

unsigned long Expression::
vectorSize()
{
  return m_expression->vectorSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
