// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExpressionResult.cc                                         (C) 2000-2018 */
/*                                                                           */
/* Contient le résultat de l´évaluation d'une expression.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/expr/ExpressionResult.h"
#include "arcane/expr/BadExpressionException.h"

//%% ARCANE_EXPR_SUPPRESS_BEGIN
#include "arcane/IVariableAccessor.h"
//%% ARCANE_EXPR_SUPPRESS_END

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//%% ARCANE_EXPR_SUPPRESS_BEGIN
ExpressionResult::
ExpressionResult(IVariable* v)
{
  if (v->dimension() != 1)
    throw BadExpressionException
    ("ExpressionResult::ExpressionResult(IVariablePrv* v)", 
     "Only variables of dimension 1 are dealt with in the expressions.");

  throw NotImplementedException(A_FUNCINFO,"building expression result after removing IVariableAccessor");
#if 0
  eDataType type = v->dataType();
  switch(type)
  {
  case DT_Real:
    {
      RealArrayView values = v->accessor()->asArrayReal();
      m_data = new ArrayVariant(values);
      break;
    }
  case DT_Int32:
    {
      Int32ArrayView values = v->accessor()->asArrayInt32();
      m_data = new ArrayVariant(values);
      break;
    }
  case DT_Int64:
    {
      Int64ArrayView values = v->accessor()->asArrayInt64();
      m_data = new ArrayVariant(values);
      break;
    }
  case DT_Real2:
    {
      Real2ArrayView values = v->accessor()->asArrayReal2();
      m_data = new ArrayVariant(values);
      break;
    }
  case DT_Real3:
    {
      Real3ArrayView values = v->accessor()->asArrayReal3();
      m_data = new ArrayVariant(values);
      break;
    }
  case DT_Real2x2:
    {
      Real2x2ArrayView values = v->accessor()->asArrayReal2x2();
      m_data = new ArrayVariant(values);
      break;
    }
  case DT_Real3x3:
    {
      Real3x3ArrayView values = v->accessor()->asArrayReal3x3();
      m_data = new ArrayVariant(values);
      break;
    }
  default:
    m_data = 0;
    String s = String("Type de variable (") + type + ") non supporté.\n";
    throw BadExpressionException("ExpressionResult::ExpressionResult(IVariable* v)",s);
    return;
  }

  Integer size = v->nbElement();
  m_own_indices.resize(size);
  for(Integer i=0 ; i<size ; ++i)
    m_own_indices[i]=i;
  m_indices = m_own_indices.view();
#endif

}
//%% ARCANE_EXPR_SUPPRESS_END

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExpressionResult::
ExpressionResult(ArrayVariant* data)
: m_data(data)
{
  Integer size = data->size();
  m_own_indices.resize(size);
  for( Integer i=0 ; i<size ; ++i)
    m_own_indices[i] = i;
  m_indices = m_own_indices.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExpressionResult::
ExpressionResult(IntegerConstArrayView indices)
: m_data(0)
{
  m_indices = indices;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExpressionResult::
~ExpressionResult()
{
  delete m_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExpressionResult::
allocate(VariantBase::eType type)
{
  if (!m_data){
    m_data = new ArrayVariant(type, m_indices.size());
  }
  else if (type != m_data->type()){
    OStringStream s;
    s() << "The result type of the expression ("
        << m_data->typeName() << ") "
        << "is not compatible with the type of the variable ("
        << VariantBase::typeName(type) << ").\n";
    throw BadExpressionException("ExpressionResult::allocate", s.str());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& 
operator<<(std::ostream& s, const ExpressionResult& x)
{
  s << "ExpressionResult [";
  if (x.m_data)
    s << "data=" << *x.m_data << ", ";
  s << "indices=[ ";
  Integer size = x.m_indices.size();
  for( Integer i=0 ; i<size ; ++i )
    s << x.m_indices[i] << " ";
  s << "]]";
  
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
