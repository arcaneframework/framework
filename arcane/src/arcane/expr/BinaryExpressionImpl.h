// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BinaryExpressionImpl.h                                      (C) 2000-2005 */
/*                                                                           */
/* Implementation d'une expression binaire.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_BINARYEXPRESSIONIMPL_H
#define ARCANE_EXPR_BINARYEXPRESSIONIMPL_H
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

class BinaryOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation d'une expression binaire
 */
class BinaryExpressionImpl
: public ExpressionImpl
{
 public:
  enum eOperationType
  {
    Add = 0,
    Substract,
    Multiply,
    Divide,
    Minimum,
    Maximum,
    Pow,
    LessThan,
    GreaterThan,
    LessOrEqualThan,
    GreaterOrEqualThan,
    Or,
    And,
    Equal,
    NbOperationType
  };
  
 public:
  BinaryExpressionImpl (IExpressionImpl* first,
                        IExpressionImpl* second,
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
  Expression m_second;
  eOperationType m_operation;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Operateur binaire generique pour les expressions.
 */
class BinaryOperator
{
 public:
  virtual ~BinaryOperator() {}
  virtual void evaluate(ExpressionResult* res, 
                        ArrayVariant* a, 
                        ArrayVariant* b)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T> class DefaultBinaryOperator
: public BinaryOperator
{
 public:
  virtual void evaluate(ArrayView<T> res, 
                        ArrayView<T> a, 
                        ArrayView<T> b)=0;

  virtual void evaluate(ExpressionResult* res, 
                        ArrayVariant* a, 
                        ArrayVariant* b)
  {
    // verification de la validite de l'operation
    if (a->type() != b->type())
      throw BadOperandException("DefaultBinaryOperator::evaluate");
    
    Integer size = res->size();
    if (size != a->size() || size != b->size())
      throw BadOperandException("DefaultBinaryOperator::evaluate");
    
    // allocation du résultat en fonction du type de a
    res->allocate(a->type());
    
    // recuperation des valeurs des operandes
    ArrayView<T> res_val;
    res->data()->value(res_val);
    ArrayView<T> a_val;
    a->value(a_val);
    ArrayView<T> b_val;
    b->value(b_val);    

    // evaluation des tableaux
    evaluate(res_val, a_val, b_val);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define DEFAULT_BINARY_OP(classname,expression) \
template<class T> \
class classname : public DefaultBinaryOperator<T> \
{ \
 public:\
  virtual void evaluate(ExpressionResult* res, \
                        ArrayVariant* a, \
                        ArrayVariant* b)\
  { DefaultBinaryOperator<T>::evaluate(res,a,b); } \
  virtual void evaluate(ArrayView<T> res, \
                        ArrayView<T> a, \
                        ArrayView<T> b) \
  { \
    Integer size=res.size(); \
    for (Integer i=0 ; i<size ; ++i) \
      expression; \
  } \
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DEFAULT_BINARY_OP(AddOperator,res[i]=a[i]+b[i])
DEFAULT_BINARY_OP(SubstractOperator,res[i]=a[i]-b[i])
DEFAULT_BINARY_OP(MultiplyOperator,res[i]=a[i]*b[i])
DEFAULT_BINARY_OP(DivideOperator,res[i]=a[i]/b[i])
DEFAULT_BINARY_OP(MinimumOperator,(a[i]<b[i])?res[i]=a[i]:res[i]=b[i])
DEFAULT_BINARY_OP(MaximumOperator,(a[i]<b[i])?res[i]=b[i]:res[i]=a[i])
DEFAULT_BINARY_OP(PowOperator,res[i]=pow(a[i],b[i]))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class BoolBinaryOperator
: public BinaryOperator
{
 public:
  virtual void evaluate(ArrayView<bool> res, 
                        ArrayView<T> a, 
                        ArrayView<T> b)=0;

  virtual void evaluate(ExpressionResult* res, 
                        ArrayVariant* a, 
                        ArrayVariant* b)
  {
    // verification de la validite de l'operation
    if (a->type() != b->type())
      throw BadOperandException("BoolBinaryOperator::evaluate");

    Integer size = res->size();
    if (size != a->size() || size != b->size())
      throw BadOperandException("BoolBinaryOperator::evaluate");

    // allocation du résultat qui doit etre booléen
    res->allocate(ArrayVariant::TBool);

    // recuperation des valeurs des operandes
    ArrayView<bool> res_val;
    res->data()->value(res_val);
    ArrayView<T> a_val;
    a->value(a_val);
    ArrayView<T> b_val;
    b->value(b_val);    

    // evaluation des tableaux
    evaluate(res_val, a_val, b_val);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define BOOL_BINARY_OP(classname,expression) \
template<class T> \
class classname : public BoolBinaryOperator<T> \
{ \
 public:\
  virtual void evaluate(ExpressionResult* res, \
                        ArrayVariant* a, \
                        ArrayVariant* b)\
  { BoolBinaryOperator<T>::evaluate(res,a,b); }\
  virtual void evaluate(ArrayView<bool> res, \
                        ArrayView<T> a, \
                        ArrayView<T> b) \
  { \
    Integer size=res.size(); \
    for (Integer i=0 ; i<size ; ++i) \
      expression; \
  } \
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BOOL_BINARY_OP(EQOperator,res[i]=(a[i]==b[i]))
BOOL_BINARY_OP(LTOperator,res[i]=(a[i]<b[i]))
BOOL_BINARY_OP(GTOperator,res[i]=(a[i]>b[i]))
BOOL_BINARY_OP(LOETOperator,res[i]=(a[i]<=b[i]))
BOOL_BINARY_OP(GOETOperator,res[i]=(a[i]>=b[i]))
BOOL_BINARY_OP(AndOperator,res[i]=(a[i]&&b[i]))
BOOL_BINARY_OP(OrOperator,res[i]=(a[i]||b[i]))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
