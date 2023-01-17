// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExpressionImpl.cc                                           (C) 2000-2014 */
/*                                                                           */
/* Implémentation d'une expression.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/Iostream.h"
#include "arcane/utils/Array.h"
#include "arcane/expr/ExpressionImpl.h"
#include "arcane/expr/OperatorMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExpressionImpl::
ExpressionImpl()
: m_op_mng(OperatorMng::instance())
, m_nb_reference(0)
, m_do_trace(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExpressionImpl::
addRef()
{
  ++m_nb_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExpressionImpl::
removeRef()
{
  --m_nb_reference;
  if (m_nb_reference==0){
    //cout << "** DELETE\n";
    delete this;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \warning L'implémentation actuelle considère que toutes les expressions
 * en paramètre sont de type réels, et l'expression de test des booléens.
 */
void ExpressionImpl::
dumpIf(IExpressionImpl* test_expr,Array<Expression>& add_exprs)
{
  UniqueArray<Expression> exprs(add_exprs.size()+1);
  exprs[0] = this;
  for( Integer i=0; i<add_exprs.size(); ++i )
    exprs[i+1] = add_exprs[i];

  Integer size = vectorSize();
  UniqueArray<bool> test_values(size);
  // Le variant appartient ensuite à résult qui le détruira
  ArrayVariant* test_variant = new ArrayVariant(test_values);
  ExpressionResult test_expr_result(test_variant);

  test_expr->apply(&test_expr_result);
  
  Integer nb_expr = exprs.size();
  SharedArray< SharedArray<Real> > display_values(nb_expr);
  for( Integer i=0; i<nb_expr; ++i ){
    display_values[i].resize(size);
    ArrayVariant* expr_variant = new ArrayVariant(display_values[i]);
    ExpressionResult expr_result(expr_variant);
    exprs[i]->apply(&expr_result);
  } 

  cout.flags(std::ios::scientific);
  std::streamsize ss = std::cout.precision();
  cout.precision(10);

  for( Integer i=0; i<size; ++i ){
    if (test_values[i]){
      cout << "valeur [" << i << "] ";
      for( Integer j=0; j<nb_expr; ++j ){
        cout << ' ' << display_values[j][i];
      }
      cout << '\n';
    }
  }
  std::cout.precision (ss);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
