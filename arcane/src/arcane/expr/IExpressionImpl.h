// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExpressionImpl.h                                           (C) 2000-2014 */
/*                                                                           */
/* Interface pour les différentes implémentations d'une expression.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_IEXPRESSIONIMPL_H
#define ARCANE_EXPR_IEXPRESSIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ExpressionResult;
class Expression;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour les différentes implémentations d'une expression.
 */
class ARCANE_EXPR_EXPORT IExpressionImpl
{
 protected:

  //! Libère les ressources. Uniquement appelé par un removeRef()
  virtual ~IExpressionImpl() {}

 public:

  virtual void assign(IExpressionImpl* expr) = 0;
  virtual void assign(IExpressionImpl* expr, IntegerConstArrayView indices) = 0;
  /*! \brief Nombre d'éléments du vecteur
   *
   * Si l'expression est un vecteur et un symbole terminal (une feuille),
   * retourne son nombre d'éléments. Sinon, retourne 0.
   */
  virtual Integer vectorSize() const =0;

  virtual void dumpIf(IExpressionImpl* test_expr,Array<Expression>& exprs) =0;
  virtual void apply(ExpressionResult* result) = 0;
  virtual void addRef() = 0;
  virtual void removeRef() = 0;
  virtual void setTrace(bool v) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
