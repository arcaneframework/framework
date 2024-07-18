// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#error "VariableExpression are no longer available. Do not include this file"
/*---------------------------------------------------------------------------*/
/* VariableExpressionImpl.h                                    (C) 2000-2024 */
/*                                                                           */
/* Expression traitant une variable.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEEXPRESSIONIMPL_H
#define ARCANE_VARIABLEEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/Expression.h"
#include "arcane/expr/ExpressionResult.h"
#include "arcane/expr/BadOperandException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class VariableOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableExpressionImpl
: public ExpressionImpl
{
 public:

  VariableExpressionImpl(IVariable* var);

 public:

  virtual void assign(IExpressionImpl* expr);
  virtual void assign(IExpressionImpl*, IntegerConstArrayView indices);
  virtual void apply(ExpressionResult* result);
  virtual Integer vectorSize() const;
  
 private:
  IVariable* m_variable;
  VariableOperator* m_op;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Operateur binaire generique pour les expressions.
 */
class VariableOperator
{
 public:
  virtual ~VariableOperator() {}
 public:
  virtual void assign(ExpressionResult* res, IVariable* var)=0;
  virtual void evaluate(ExpressionResult* res, IVariable* var)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class VariableOperatorT
: public VariableOperator
{
 public:
  virtual ~VariableOperatorT() {}
 public:
  virtual void assign(ExpressionResult* res,IVariable* var)
  {
    // Notons que la taille de la variable peut être plus importante
    // que celle du résultat suivant les indices retenus (cf WhereExpression)
    Integer size = res->size();

    // recuperation des valeurs du resultat et de la variable
    ArrayView<T> res_val;
    res->data()->value(res_val);
    ExpressionResult var_res(var);
    ArrayView<T> var_val;
    var_res.data()->value(var_val);
    IntegerConstArrayView res_indices = res->indices();

    for( Integer i=0 ; i<size ; ++i)
      var_val[res_indices[i]] = res_val[i];
  }  

  virtual void evaluate(ExpressionResult* res, IVariable* var)
  {
    // Notons que la taille de la variable peut être plus importante
    // que celle du résultat suivant les indices retenus (cf WhereExpression)
    Integer size = res->size();
    if (size > var->nbElement())
      throw BadOperandException("VariableOperatorT::evaluate");

    // allocation du résultat en fonction du type de la variable
    VariantBase::eType type = VariantBase::fromDataType(var->dataType());
    res->allocate(type);

    // recuperation des valeurs des operandes
    ArrayView<T> res_val;
    res->data()->value(res_val);
    ExpressionResult var_res(var);
    ArrayView<T> var_val;
    var_res.data()->value(var_val);
    IntegerConstArrayView res_indices = res->indices();

    for( Integer i=0 ; i<size ; ++i)
      res_val[i] = var_val[res_indices[i]];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
