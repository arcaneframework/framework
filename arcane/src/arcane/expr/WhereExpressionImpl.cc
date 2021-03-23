// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WhereExpressionImpl.cc                                      (C) 2000-2004 */
/*                                                                           */
/* Implémentation d'une expression conditionnelle.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/expr/WhereExpressionImpl.h"
#include "arcane/expr/OperatorMng.h"
#include "arcane/expr/BadOperationException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

WhereExpressionImpl::
WhereExpressionImpl(IExpressionImpl* test,IExpressionImpl* iftrue,
                    IExpressionImpl* iffalse)
: ExpressionImpl()
, m_test(test)
, m_iftrue(iftrue)
, m_iffalse(iffalse) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void WhereExpressionImpl::
apply(ExpressionResult* result)
{
  // cerr << ">> BEGIN WHERE EXPRESSION [" << *result << "]\n";

  // evaluation du test
  IntegerConstArrayView indices = result->indices();
  ExpressionResult test_op(indices);
  m_test->apply(&test_op);
  ArrayView<bool> test_values;
  test_op.data()->value(test_values);
  Integer s = test_values.size();
  Integer nb_true=0;  
  Integer nb_false=0;
  for(Integer i=0 ; i<s ; ++i) 
    (test_values[i]==0.) ? ++nb_false : ++nb_true;
  UniqueArray<Integer> true_indices(nb_true);
  UniqueArray<Integer> false_indices(nb_false);
  Integer true_i = 0;
  Integer false_i = 0;
  for(Integer i=0 ; i<s ; ++i) 
    (test_values[i]!=0.) ? true_indices[true_i++]=indices[i] 
    : false_indices[false_i++]=indices[i];
  
  /*
    cerr << "Expression where."<< endl;
    cerr << "   false indices = [ ";
    for(Integer i=0 ; i<nb_false ; ++i)
    cerr << false_indices[i] << " ";
    cerr << "]."<< endl;
    cerr << "   true indices = [ ";
    for(Integer i=0 ; i<nb_true ; ++i)
    cerr << true_indices[i] << " ";
    cerr << "]."<< endl;
  */

  // evaluation de l'expression en fonction du resultat du test
  ExpressionResult iftrue_op(true_indices);
  m_iftrue->apply(&iftrue_op);
  ExpressionResult iffalse_op(false_indices);
  m_iffalse->apply(&iffalse_op);

  // recuperation des resultat
  VariantBase::eType type = iftrue_op.data()->type();
  WhereOperator* op = m_op_mng->find(this, type);
  if (!op)
    throw BadOperationException("WhereExpressionImpl::apply","",type);

  op->evaluate(result, test_op.data(), iftrue_op.data(), iffalse_op.data());

  // cerr << "<< END WHERE EXPRESSION [" << *result << "]\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
