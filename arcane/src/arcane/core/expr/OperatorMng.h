// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OperatorMng.h                                               (C) 2000-2004 */
/*                                                                           */
/* Stocke tous les types d'opérateur possibles sur les expressions.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_OPERATORMNG_H
#define ARCANE_UTILS_OPERATORMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/expr/UnaryExpressionImpl.h"
#include "arcane/expr/BinaryExpressionImpl.h"
#include "arcane/expr/WhereExpressionImpl.h"
#include "arcane/expr/LitteralExpressionImpl.h"

//%% ARCANE_EXPR_SUPPRESS_BEGIN
#include "arcane/VariableExpressionImpl.h"
//%% ARCANE_EXPR_SUPPRESS_END

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Stocke tous les types d'operateur possibles sur les expressions.
 */
class ARCANE_EXPR_EXPORT OperatorMng
{
 protected:
  OperatorMng();
  ~OperatorMng();

 public:
  static OperatorMng* instance();

 public:
  UnaryOperator* find(UnaryExpressionImpl* e, 
                      VariantBase::eType type, 
                      UnaryExpressionImpl::eOperationType operation);
  
  BinaryOperator* find(BinaryExpressionImpl* e, 
                       VariantBase::eType type, 
                       BinaryExpressionImpl::eOperationType operation);
  
  WhereOperator* find(WhereExpressionImpl* e, 
                      VariantBase::eType type);
  
  LitteralOperator* find(LitteralExpressionImpl* e, 
                         VariantBase::eType type);
  
//%% ARCANE_EXPR_SUPPRESS_BEGIN
  VariableOperator* find(VariableExpressionImpl* e, 
                         VariantBase::eType type);
//%% ARCANE_EXPR_SUPPRESS_END

 private:
  static OperatorMng* m_instance;

 private:
  typedef std::map<VariantBase::eType, UnaryOperator*, 
    std::less<VariantBase::eType> > UnaryOpMap;
  UnaryOpMap m_unary_op[UnaryExpressionImpl::NbOperationType];

  typedef std::map<VariantBase::eType, BinaryOperator*, 
    std::less<Integer> > BinaryOpMap;
  BinaryOpMap m_binary_op[BinaryExpressionImpl::NbOperationType];

  typedef std::map<VariantBase::eType, WhereOperator*, 
    std::less<VariantBase::eType> > WhereOpMap;
  WhereOpMap m_where_op;

  typedef std::map<VariantBase::eType, LitteralOperator*, 
    std::less<VariantBase::eType> > LitteralOpMap;
  LitteralOpMap m_litteral_op;

//%% ARCANE_EXPR_SUPPRESS_BEGIN
  typedef std::map<VariantBase::eType, VariableOperator*, 
    std::less<VariantBase::eType> > VariableOpMap;
  VariableOpMap m_variable_op;
//%% ARCANE_EXPR_SUPPRESS_END
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
