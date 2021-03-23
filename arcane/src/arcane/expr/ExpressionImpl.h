// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExpressionImpl.h                                            (C) 2000-2014 */
/*                                                                           */
/* Implémentation d'une expression.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_EXPRESSIONIMPL_H
#define ARCANE_EXPR_EXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/expr/IExpressionImpl.h"
#include "arcane/expr/Expression.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OperatorMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de l'implémentation d'une expression.
 */
class ARCANE_EXPR_EXPORT ExpressionImpl
: public IExpressionImpl
{
 public:
  ExpressionImpl();

 public:

  virtual void addRef();
  virtual void removeRef();
  virtual void setTrace(bool v){ m_do_trace = v; }
  virtual void dumpIf(IExpressionImpl* test_expr,Array<Expression>& exprs);

 protected:

  OperatorMng* m_op_mng;
  bool isTraceMode() const { return m_do_trace; }

 private:

  Integer m_nb_reference; //!< Nombre de références
  bool m_do_trace;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
