// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnaryExpressionImpl.cc                                      (C) 2000-2007 */
/*                                                                           */
/* Implémentation d'une expression unaire.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/expr/UnaryExpressionImpl.h"
#include "arcane/expr/OperatorMng.h"
#include "arcane/expr/BadOperationException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnaryExpressionImpl::
UnaryExpressionImpl (IExpressionImpl* first,eOperationType operation)
: ExpressionImpl()
, m_first(first)
, m_operation(operation) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String UnaryExpressionImpl::
operationName(eOperationType type)
{
  switch(type)
  {
  case UnarySubstract: return "UnarySubstract";
  case Inverse: return "Inverse";
  case Acos: return "Acos";
  case Asin: return "Asin";
  case Atan: return "Atan";
  case Ceil: return "Ceil";
  case Cos: return "Cos";
  case Cosh: return "Cosh";
  case Exp: return "Exp";
  case Fabs: return "Fabs";
  case Floor: return "Floor";
  case Log: return "Log";
  case Log10: return "Log10";
  case Sin: return "Sin";
  case Sinh: return "Sinh";
  case Sqrt: return "Sqrt";
  case Tan: return "Tan";
  case Tanh: return "Tanh";
  default: return "Unknown";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnaryExpressionImpl::
apply(ExpressionResult* result)
{
  /*
    cerr << ">> BEGIN UNARY EXPRESSION " 
    << operationName() << " [" << *result << "]\n";
  */

  // calcul des expressions gauche et droite
  ExpressionResult first_op(result->indices());
  m_first->apply(&first_op);

  // recherche de l'operateur en fonction du type attendu en resultat
  VariantBase::eType type = first_op.data()->type();
  UnaryOperator* op = m_op_mng->find(this, type, m_operation);
  if (!op)
    throw BadOperationException("UnaryExpressionImpl::apply", operationName(), type);

  op->evaluate(result, first_op.data());

  /*
    cerr << "<< END UNARY EXPRESSION " 
    << operationName() << " [" << *result << "]\n";
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
