// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BinaryExpressionImpl.cc                                     (C) 2000-2007 */
/*                                                                           */
/* Implementation d'une expression binaire.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/expr/BinaryExpressionImpl.h"
#include "arcane/expr/OperatorMng.h"
#include "arcane/expr/BadOperationException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BinaryExpressionImpl::
BinaryExpressionImpl(IExpressionImpl* first,IExpressionImpl* second,
                     eOperationType operation)
: ExpressionImpl()
, m_first(first)
, m_second(second)
, m_operation(operation) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String BinaryExpressionImpl::
operationName(eOperationType type)
{
  switch(type)
  {
  case Add: return "Add";
  case Substract: return "Substract";
  case Multiply: return "Multiply";
  case Divide: return "Divide";
  case Minimum: return "Minimum";
  case Maximum: return "Maximum";
  case Pow: return "Pow";
  case LessThan: return "LessThan";
  case GreaterThan: return "GreaterThan";
  case LessOrEqualThan: return "LessOrEqualThan";
  case GreaterOrEqualThan: return "GreaterOrEqualThan";
  default: return "Unknown";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BinaryExpressionImpl::
apply(ExpressionResult* result)
{
 /*
    cerr << ">> BEGIN BINARY EXPRESSION " 
    << operationName() << " [" << *result << "]\n";
  */

  // calcul des expressions gauche et droite
  ExpressionResult first_op(result->indices());
  m_first->apply(&first_op);
  ExpressionResult second_op(result->indices());
  m_second->apply(&second_op);

  // recherche de l'operateur en fonction du type attendu en resultat
  VariantBase::eType type = first_op.data()->type();
  BinaryOperator* op = m_op_mng->find(this, type, m_operation);
  if (!op)
    throw BadOperationException("BinaryExpressionImpl::apply",
                                operationName(),type);

  op->evaluate(result, first_op.data(), second_op.data());

  /*
    cerr << "<< END BINARY EXPRESSION " 
    << operationName() << " [" << *result << "]\n";
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

