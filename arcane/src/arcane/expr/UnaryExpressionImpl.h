// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnaryExpressionImpl.h                                       (C) 2000-2006 */
/*                                                                           */
/* Implémentation d'une expression unaire.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_UNARYEXPRESSIONIMPL_H
#define ARCANE_EXPR_UNARYEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Convert.h"
#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/Expression.h"
#include "arcane/expr/ExpressionResult.h"
#include "arcane/expr/BadOperandException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnaryOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation d'une expression unaire
 */
class ARCANE_EXPR_EXPORT UnaryExpressionImpl
: public ExpressionImpl
{
 public:
  enum eOperationType
  {
    UnarySubstract = 0,
    Inverse = 1,
    Acos = 2,
    Asin = 3,
    Atan = 4,
    Ceil = 5,
    Cos = 6,
    Cosh = 7,
    Exp = 8,
    Fabs = 9,
    Floor = 10,
    Log = 11,
    Log10 = 12,
    Sin = 13,
    Sinh = 14,
    Sqrt = 15,
    Tan = 16,
    Tanh = 17,
    NbOperationType = 18
  };
  
 public:
  UnaryExpressionImpl (IExpressionImpl* first,
                       eOperationType operation);

 public:
  virtual void assign(IExpressionImpl*) {}
  virtual void assign(IExpressionImpl*, IntegerConstArrayView) {}
  virtual void apply(ExpressionResult* result);
  virtual Integer vectorSize() const { return 0; }
  String operationName() const { return operationName(m_operation); }
  static String operationName(eOperationType type);

 private:
  Expression m_first;
  eOperationType m_operation;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Operateur unaire generique pour les expressions.
 */
class UnaryOperator
{
 public:
  virtual ~UnaryOperator() {}
  virtual void evaluate(ExpressionResult* res, ArrayVariant* a)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class DefaultUnaryOperator
: public UnaryOperator
{
 public:
  virtual void evaluate(ArrayView<T> res, 
                        ArrayView<T> a)=0;

  virtual void evaluate(ExpressionResult* res, ArrayVariant* a)
  {
    // verification de la validite de l'operation
    Integer size = res->size();
    if (size != a->size())
      throw BadOperandException("DefaultUnaryOperator::evaluate");

    // allocation du résultat en fonction du type de a
    res->allocate(a->type());

    // recuperation des valeurs des operandes
    ArrayView<T> res_val;
    res->data()->value(res_val);
    ArrayView<T> a_val;
    a->value(a_val);

    // evaluation des tableaux
    evaluate(res_val, a_val);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define DEFAULT_UNARY_OP(classname,expression) \
template<class T> \
class classname : public DefaultUnaryOperator<T> \
{ \
 public:\
  virtual void evaluate(ExpressionResult* res, ArrayVariant* a) \
  { DefaultUnaryOperator<T>::evaluate(res,a); } \
\
  virtual void evaluate(ArrayView<T> res, \
                        ArrayView<T> a) \
  { \
    Integer size=res.size(); \
    for (Integer i=0 ; i<size ; ++i) \
      expression; \
  } \
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DEFAULT_UNARY_OP(UnarySubstractOperator,res[i]=-a[i])
DEFAULT_UNARY_OP(InverseOperator,res[i]=1/a[i])
DEFAULT_UNARY_OP(AcosOperator,res[i]=acos(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(AsinOperator,res[i]=asin(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(AtanOperator,res[i]=atan(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(CeilOperator,res[i]=ceil(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(CosOperator,res[i]=cos(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(CoshOperator,res[i]=cosh(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(ExpOperator,res[i]=exp(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(FabsOperator,res[i]=fabs(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(FloorOperator,res[i]=floor(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(LogOperator,res[i]=log(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(Log10Operator,res[i]=log10(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(SinOperator,res[i]=sin(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(SinhOperator,res[i]=sinh(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(SqrtOperator,res[i]=sqrt(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(TanOperator,res[i]=tan(Convert::toDouble(a[i])))
DEFAULT_UNARY_OP(TanhOperator,res[i]=tanh(Convert::toDouble(a[i])))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
