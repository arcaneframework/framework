// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LitteralExpressionImpl.h                                    (C) 2000-2022 */
/*                                                                           */
/* Implémentation d'une expression littérale contenant un scalaire.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_LITTERALEXPRESSIONIMPL_H
#define ARCANE_EXPR_LITTERALEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/core/datatype/ScalarVariant.h"

#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/Expression.h"
#include "arcane/expr/ExpressionResult.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LitteralOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implementation d'une expression litterale contenant un scalaire.
 */
class LitteralExpressionImpl
: public ExpressionImpl
{
 public:

  explicit LitteralExpressionImpl (const ScalarVariant& value);

 public:

  virtual void assign(IExpressionImpl*) {}
  virtual void assign(IExpressionImpl*, IntegerConstArrayView) {}
  virtual void apply(ExpressionResult* result);
  virtual Integer vectorSize() const { return 0; }

 private:

  ScalarVariant m_value; 
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Operateur de cast generique pour les litteraux.
 */
class LitteralOperator
{
 public:

  virtual ~LitteralOperator() {}

 public:

  virtual void evaluate(ExpressionResult* res, ScalarVariant& a)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class LitteralOperatorT
: public LitteralOperator
{
 public:

  void evaluate(ExpressionResult* res, ScalarVariant& a) override
  {
    // allocation du résultat en fonction du type du variant
    res->allocate(a.type());

    // recuperation des valeurs des operandes
    ArrayView<T> res_val;
    res->data()->value(res_val);
    T a_val;
    a.value(a_val);

    Integer size = res->data()->size();
    for( Integer i=0 ; i<size ; ++i)
      res_val[i] = a_val;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
