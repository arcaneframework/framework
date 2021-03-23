// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WhereExpressionImpl.h                                       (C) 2000-2004 */
/*                                                                           */
/* Implémentation d'une expression conditionnelle.                           */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class WhereOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation d'une expression binaire
 */
class ARCANE_EXPR_EXPORT WhereExpressionImpl
: public ExpressionImpl
{
 public:
  WhereExpressionImpl (IExpressionImpl* test,
                       IExpressionImpl* iftrue,
                       IExpressionImpl* iffalse);

 public:
  virtual void assign(IExpressionImpl*) {}
  virtual void assign(IExpressionImpl*, IntegerConstArrayView) {}
  virtual void apply(ExpressionResult* result);
  virtual Integer vectorSize() const { return 0; }

 private:
  Expression m_test;      //!< Expression de test
  Expression m_iftrue;    //!< Expression évaluée lorsque le test est positif
  Expression m_iffalse;   //!< Expression évaluée lorsque le test est negatif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Operateur generique pour les expressions conditionnnelle.
 */
class WhereOperator
{
 public:
  virtual ~WhereOperator(){}
  virtual void evaluate(ExpressionResult* res, 
                        ArrayVariant* test,
                        ArrayVariant* iftrue, 
                        ArrayVariant* iffalse) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
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
    // verification de la validite de l'operation
    if (test->type() != ArrayVariant::TBool)
      throw BadOperandException("WhereOperatorT::evaluate");

    Integer size = res->size();
    if (size != test->size())
      throw BadOperandException("WhereOperatorT::evaluate");

    if (iftrue->type() || iffalse->type())
      throw BadOperandException("WhereOperatorT::evaluate");

    // allocation du résultat en fonction du type du résultat du if
    res->allocate(iftrue->type());

    // recuperation des valeurs des operandes
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
    for (Integer i=0 ; i<size ; ++i)
      test_val[i] ? res_val[i] = iftrue_val[true_i++]
      : res_val[i] = iffalse_val[false_i++];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
