// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableExpressionImpl.cc                                   (C) 2000-2004 */
/*                                                                           */
/* Expression handling a variable.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/core/IVariableAccessor.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/VariableExpressionImpl.h"

#include "arcane/expr/OperatorMng.h"
#include "arcane/expr/BadOperationException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableExpressionImpl::
VariableExpressionImpl(IVariable* var)
: ExpressionImpl()
, m_variable(var)
{
  VariantBase::eType type = VariantBase::fromDataType(m_variable->dataType());
  m_op = m_op_mng->find(this, type);
  if (!m_op)
    throw BadOperationException("VariableExpressionImpl::VariableExpressionImpl", "", type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableExpressionImpl::
assign(IExpressionImpl* expr)
{
  ExpressionResult result(m_variable);
  expr->apply(&result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableExpressionImpl::
assign(IExpressionImpl* expr, IntegerConstArrayView indices)
{
  ExpressionResult result(indices);
  result.allocate(VariantBase::fromDataType(m_variable->dataType()));
  expr->apply(&result);
  m_op->assign(&result, m_variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableExpressionImpl::
apply(ExpressionResult* result)
{
  m_op->evaluate(result, m_variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer VariableExpressionImpl::
vectorSize() const
{
  if (m_variable->dimension() != 1)
    return 0;
  return m_variable->nbElement();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
