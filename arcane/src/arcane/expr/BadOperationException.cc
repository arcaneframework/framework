// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadOperationException.cc                                    (C) 2000-2016 */
/*                                                                           */
/* Exception sur une opération des expressions.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"

#include "arcane/expr/BadOperationException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadOperationException::
BadOperationException(const String& where,const String& operation_name,
                      VariantBase::eType operand_type)
: Exception("BadOperation",where)
, m_operation_name(operation_name)
, m_operand_type(operand_type)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadOperationException::
BadOperationException(const BadOperationException& ex)
: Exception(ex)
, m_operation_name(ex.m_operation_name)
, m_operand_type(ex.m_operand_type)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadOperationException::
explain(std::ostream& m) const
{
  m << "Operation " << m_operation_name
    << " non définie pour le type " 
    << VariantBase::typeName(m_operand_type)
    << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


