// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OperatorMng.cc                                              (C) 2000-2003 */
/*                                                                           */
/* Stocke tous les types d'opérateur possibles sur les expressions.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/expr/OperatorMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OperatorMng* OperatorMng::m_instance = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define CREATE_ALL_OP(map,name) \
  map .insert(name##OpMap::value_type \
               (VariantBase::TReal, new name##OperatorT<Real>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TInt64, new name##OperatorT<Int64>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TInt32, new name##OperatorT<Int32>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TBool, new name##OperatorT<bool>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TReal2, new name##OperatorT<Real2>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TReal3, new name##OperatorT<Real3>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TReal2x2, new name##OperatorT<Real2x2>())); \
  map .insert(name##OpMap::value_type \
               (VariantBase::TReal3x3, new name##OperatorT<Real3x3>()));

#define CREATE_OP(map,var,maptype,varname,vartype,operator) \
  map[var].insert( maptype ::value_type(VariantBase::varname, \
  new operator##Operator<vartype>()));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OperatorMng::
OperatorMng()
: m_unary_op()
{
  //
  // construction des opérateurs unaires
  //
  UnaryExpressionImpl::eOperationType ut;
  ut = UnaryExpressionImpl::UnarySubstract;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,UnarySubstract)
  ut = UnaryExpressionImpl::Inverse;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Inverse)
  ut = UnaryExpressionImpl::Acos;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Acos)
  ut = UnaryExpressionImpl::Asin;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Asin)
  ut = UnaryExpressionImpl::Atan;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Atan)
  ut = UnaryExpressionImpl::Ceil;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Ceil)
  ut = UnaryExpressionImpl::Cos;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Cos)
  ut = UnaryExpressionImpl::Cosh;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Cosh)
  ut = UnaryExpressionImpl::Exp;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Exp)
  ut = UnaryExpressionImpl::Fabs;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Fabs)
  ut = UnaryExpressionImpl::Floor;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Floor)
  ut = UnaryExpressionImpl::Log;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Log)
  ut = UnaryExpressionImpl::Log10;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Log10)
  ut = UnaryExpressionImpl::Sin;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Sin)
  ut = UnaryExpressionImpl::Sinh;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Sinh)
  ut = UnaryExpressionImpl::Sqrt;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Sqrt)
  ut = UnaryExpressionImpl::Tan;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Tan)
  ut = UnaryExpressionImpl::Tanh;
  CREATE_OP(m_unary_op,ut,UnaryOpMap,TReal,Real,Tanh)

  //
  // construction des opérateurs binaires
  //
  BinaryExpressionImpl::eOperationType bt;
  // add
  bt = BinaryExpressionImpl::Add;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Add)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,Add)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,Add)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,Add)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,Add)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2x2,Real2x2,Add)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3x3,Real3x3,Add)
  // substract
  bt = BinaryExpressionImpl::Substract;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Substract)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,Substract)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,Substract)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,Substract)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,Substract)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2x2,Real2x2,Substract)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3x3,Real3x3,Substract)
  // multiply
  bt = BinaryExpressionImpl::Multiply;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Multiply)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,Multiply)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,Multiply)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,Multiply)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,Multiply)
  //CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2x2,Real2x2,Multiply)
  //CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3x3,Real3x3,Multiply)
  // multiply
  bt = BinaryExpressionImpl::Divide;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Divide)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,Divide)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,Divide)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,Divide)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,Divide)
  //CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2x2,Real2x2,Divide)
  //CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3x3,Real3x3,Divide)
  // minimum (operator< defini sur Real2 et Real3)
  bt = BinaryExpressionImpl::Minimum;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Minimum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,Minimum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,Minimum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,Minimum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,Minimum)
  // maximum (operator> defini sur Real2 et Real3)
  bt = BinaryExpressionImpl::Maximum;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Maximum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,Maximum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,Maximum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,Maximum)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,Maximum)
  // pow
  bt = BinaryExpressionImpl::Pow;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,Pow)
  // equal than (defini sur Real2 et Real3)
  bt = BinaryExpressionImpl::Equal;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,EQ)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,EQ)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,EQ)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,EQ)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,EQ)
  // less than (defini sur Real2 et Real3)
  bt = BinaryExpressionImpl::LessThan;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,LT)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,LT)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,LT)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal2,Real2,LT)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal3,Real3,LT)
  // less or equal than
  bt = BinaryExpressionImpl::LessOrEqualThan;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,LOET)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,LOET)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,LOET)
  // greater than
  bt = BinaryExpressionImpl::GreaterThan;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,GT)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,GT)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,GT)
  // greater or equal than
  bt = BinaryExpressionImpl::GreaterOrEqualThan;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TReal,Real,GOET)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt32,Integer,GOET)
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TInt64,Integer,GOET)
  // or
  bt = BinaryExpressionImpl::Or;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TBool,bool,Or)
  // and
  bt = BinaryExpressionImpl::And;
  CREATE_OP(m_binary_op,bt,BinaryOpMap,TBool,bool,And)

  //
  // construction des opérateurs "where"
  //
  CREATE_ALL_OP(m_where_op,Where)
//%% ARCANE_EXPR_SUPPRESS_BEGIN
  CREATE_ALL_OP(m_variable_op,Variable)
//%% ARCANE_EXPR_SUPPRESS_END
  CREATE_ALL_OP(m_litteral_op,Litteral)
}

OperatorMng::
~OperatorMng()
{
  for (int i=0 ; i<UnaryExpressionImpl::NbOperationType ; i++)
  {
    UnaryOpMap& m = m_unary_op[i];
    for (UnaryOpMap::iterator it=m.begin() ; it!=m.end() ; it++)
      delete (*it).second;
  }

  for (int i=0 ; i<BinaryExpressionImpl::NbOperationType ; i++)
  {
    BinaryOpMap& m = m_binary_op[i];
    for (BinaryOpMap::iterator it=m.begin() ; it!=m.end() ; it++)
      delete (*it).second;
  }

  for (WhereOpMap::iterator it=m_where_op.begin();
       it!=m_where_op.end();
       it++)
    delete (*it).second;

//%% ARCANE_EXPR_SUPPRESS_BEGIN
  for (VariableOpMap::iterator it=m_variable_op.begin();
       it!=m_variable_op.end();
       it++)
    delete (*it).second;
//%% ARCANE_EXPR_SUPPRESS_END

  for (LitteralOpMap::iterator it=m_litteral_op.begin();
       it!=m_litteral_op.end();
       it++)
    delete (*it).second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OperatorMng* OperatorMng::
instance()
{
  if (!m_instance)
    m_instance = new OperatorMng();
  return m_instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnaryOperator* OperatorMng::
find(UnaryExpressionImpl*, VariantBase::eType type, 
     UnaryExpressionImpl::eOperationType operation)
{
  UnaryOpMap::iterator it=m_unary_op[operation].find(type);
  if (it == m_unary_op[operation].end())
    return 0;
  else
    return (*it).second;
}

BinaryOperator* OperatorMng::
find(BinaryExpressionImpl*, VariantBase::eType type, 
     BinaryExpressionImpl::eOperationType operation)
{
  BinaryOpMap::iterator it=m_binary_op[operation].find(type);
  if (it == m_binary_op[operation].end())
    return 0;
  else
    return (*it).second;
}

WhereOperator* OperatorMng::
find(WhereExpressionImpl*, VariantBase::eType type)
{
  WhereOpMap::iterator it=m_where_op.find(type);
  if (it == m_where_op.end())
    return 0;
  else
    return (*it).second;
}

LitteralOperator* OperatorMng::
find(LitteralExpressionImpl*, VariantBase::eType type)
{
  LitteralOpMap::iterator it=m_litteral_op.find(type);
  if (it == m_litteral_op.end())
    return 0;
  else
    return (*it).second;
}

//%% ARCANE_EXPR_SUPPRESS_BEGIN
VariableOperator* OperatorMng::
find(VariableExpressionImpl*, VariantBase::eType type)
{
  VariableOpMap::iterator it=m_variable_op.find(type);
  if (it == m_variable_op.end())
    return 0;
  else
    return (*it).second;
}
//%% ARCANE_EXPR_SUPPRESS_END

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
