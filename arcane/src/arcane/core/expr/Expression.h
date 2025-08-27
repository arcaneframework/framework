// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Expression.h                                                (C) 2000-2014 */
/*                                                                           */
/* Référence à une expression.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_EXPRESSION_H
#define ARCANE_EXPR_EXPRESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/core/datatype/ScalarVariant.h"

#include "arcane/expr/IExpressionImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IExpressionImpl;
class ExpressionResult;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence à une expression.
 *
 Cette classe sert juste pour maintenir une référence à une expression
 (IExpression). Elle est utilisée dans les wrapper pour garantir qu'une
 expression allouée est bien détruite lorsqu'elle n'est plus utilisée.
 */
class ARCANE_EXPR_EXPORT Expression
{
 public:

  //! Constructeur par défaut.
  Expression();

  Expression(IExpressionImpl*);

  explicit Expression(Real v);

  /*! \brief Constructeur de recopie.
   * Le constructeur est protégé pour éviter de dupliquer par erreur une
   * instance.
   */
  Expression(const Expression& expr);

  void operator=(const Expression& expr);
  virtual ~Expression();

 public:

  void assign(const Expression& expr);
  void assign(const Expression& expr,const Array<Integer>& indices);
  void assign(Real val);

  void apply(ExpressionResult* result);

  void dumpIf(const Expression& test_expr,Array<Expression>& exprs);
  void dumpIf(const Expression& test_expr);

  IExpressionImpl* operator->() const;

  Expression operator-();
  Expression inverse();

  Expression acos();
  Expression asin();
  Expression atan();
  Expression ceil();
  Expression cos();
  Expression cosh();
  Expression exp();
  Expression fabs();
  Expression floor();
  Expression log();
  Expression log10();
  Expression sin();
  Expression sinh();
  Expression sqrt();
  Expression tan();
  Expression tanh();

  Expression operator+(Expression ex1);
  Expression operator-(Expression ex1);
  Expression operator*(Expression ex1);
  Expression operator/(Expression ex1);

  Expression eq(Expression ex1);
  Expression lt(Expression ex1);
  Expression gt(Expression ex1);
  Expression lte(Expression ex1);
  Expression gte(Expression ex1);

  Expression eor(Expression ex1);
  Expression eand(Expression ex1);

  Expression operator+(Real a)
    { return this->operator+(literal(a)); }
  Expression operator-(Real a)
    { return this->operator-(literal(a)); }
  Expression operator*(Real a)
    { return this->operator*(literal(a)); }
  Expression operator/(Real a)
    { return this->operator/(literal(a)); }

  Expression eq(Real a)
    { return this->eq(literal(a)); }
  Expression lt(Real a)
    { return this->lt(literal(a)); }
  Expression gt(Real a)
    { return this->gt(literal(a)); }
  Expression lte(Real a)
    { return this->lte(literal(a)); }
  Expression gte(Real a)
    { return this->gte(literal(a)); }

  Expression eand(Real a)
    { return this->eand(literal(a)); }
  Expression eor(Real a)
    { return this->eor(literal(a)); }

  Expression ifelse(Expression ex1, Expression ex2);
  Expression ifelse(Real ex1, Real ex2)
    { return this->ifelse(literal(ex1), literal(ex2)); }
  Expression ifelse(Expression ex1, Real ex2)
    { return this->ifelse(ex1, literal(ex2)); }
  Expression ifelse(Real ex1, Expression ex2)
    { return this->ifelse(literal(ex1), ex2); }

  Expression minimum(Expression v);
  Expression minimum(Real v)
    { return this->minimum(literal(v)); }

  Expression maximum(Expression v);
  Expression maximum(Real v)
    { return this->maximum(literal(v)); }

  Expression pow(Expression v);
  Expression pow(Real v)
    { return this->pow(literal(v)); }
  
  Expression literal(Real v);
  
  void setTrace(bool v);

  unsigned long vectorSize();

 private:
  IExpressionImpl* m_expression;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
