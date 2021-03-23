// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExpressionImpl.cc                                      (C) 2000-2005 */
/*                                                                           */
/* Expression traitant un tableau.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/datatype/ArrayVariant.h"

#include "arcane/expr/ArrayExpressionImpl.h"
#include "arcane/expr/OperatorMng.h"
#include "arcane/expr/BadOperationException.h"

#include "arcane/MathUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Operateur binaire generique pour les expressions.
 */
class ArrayOperator
{
 public:
  virtual ~ArrayOperator() {}
 public:
  virtual void assign(ExpressionResult* res, ArrayVariant* var)=0;
  virtual void evaluate(ExpressionResult* res, ArrayVariant* var)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class ArrayOperatorT
: public ArrayOperator
{
 public:
  virtual void assign(ExpressionResult* res,ArrayVariant* var)
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

  virtual void evaluate(ExpressionResult* res,ArrayVariant* var)
  {
    // Notons que la taille de la variable peut être plus importante
    // que celle du résultat suivant les indices retenus (cf WhereExpression)
    Integer size = res->size();
    Integer vsize = var->size();
    cerr << "** SIZE res=" << size << " var=" << vsize << " res=" << res << '\n';
    Integer max_size = math::min(size,vsize);
    //if (size > var->size())
    //throw BadOperandException("VariableOperatorT::evaluate");

    // allocation du résultat en fonction du type de la variable
    VariantBase::eType type = var->type();
    res->allocate(type);

    // recuperation des valeurs des operandes
    ArrayView<T> res_val;
    res->data()->value(res_val);
    ExpressionResult var_res(var);
    ArrayView<T> var_val;
    var_res.data()->value(var_val);
    IntegerConstArrayView res_indices = res->indices();

    for( Integer i=0 ; i<max_size ; ++i)
      res_val[i] = var_val[res_indices[i]];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayExpressionImpl::
ArrayExpressionImpl(ArrayVariant* variant)
: ExpressionImpl()
, m_variant(variant)
, m_op(0)
{
  switch(variant->type()){
  case VariantBase::TReal:
    m_op = new ArrayOperatorT<Real>();
    break;
  default:
    throw BadOperationException("ArrayExpressionImpl::ArrayExpressionImpl",
				"bad type",variant->type());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayExpressionImpl::
~ArrayExpressionImpl()
{
  //TODO Le delete fait planter...
  //delete m_variant;
  delete m_op;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayExpressionImpl::
assign(IExpressionImpl* expr)
{
  ExpressionResult result(m_variant);
  expr->apply(&result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayExpressionImpl::
assign(IExpressionImpl* expr,ConstArrayView<Integer> indices)
{
  ExpressionResult result(indices);
  result.allocate(m_variant->type());
  expr->apply(&result);
  m_op->assign(&result, m_variant);  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayExpressionImpl::
apply(ExpressionResult* result)
{
  m_op->evaluate(result, m_variant);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ArrayExpressionImpl::
vectorSize() const
{
  return m_variant->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

