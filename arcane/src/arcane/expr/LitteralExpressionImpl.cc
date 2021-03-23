// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LitteralExpressionImpl.cc                                   (C) 2000-2004 */
/*                                                                           */
/* Implémentation d'une expression littérale contenant un scalaire.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/expr/LitteralExpressionImpl.h"
#include "arcane/expr/OperatorMng.h"
#include "arcane/expr/BadOperationException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LitteralExpressionImpl::
LitteralExpressionImpl (ScalarVariant value)
: ExpressionImpl()
, m_value(value)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LitteralExpressionImpl::
apply(ExpressionResult* result)
{
  // cerr << ">> BEGIN LITTERAL EXPRESSION [" << *result << "]\n";

  // recherche de l'operateur en fonction du type attendu en resultat
  ScalarVariant::eType type = m_value.type();
  LitteralOperator* op = m_op_mng->find(this, type);
  if (!op)
    throw BadOperationException("LitteralExpressionImpl::apply","",type);

  op->evaluate(result, m_value);

  // cerr << "<< END LITTERAL EXPRESSION [" << *result << "]\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

