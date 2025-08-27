// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExpressionResult.h                                          (C) 2000-2004 */
/*                                                                           */
/* Contient le résultat de l´évaluation d'une expression.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_EXPRESSIONRESULT_H
#define ARCANE_EXPR_EXPRESSIONRESULT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcaneGlobal.h"

//%% ARCANE_EXPR_SUPPRESS_BEGIN
#include "arcane/core/IVariable.h"
//%% ARCANE_EXPR_SUPPRESS_END

#include "arcane/core/datatype/ArrayVariant.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Type de base polymorphe d'une expression.
 */
class ARCANE_EXPR_EXPORT ExpressionResult
{
 public:
//%% ARCANE_EXPR_SUPPRESS_BEGIN
  ExpressionResult(IVariable* v);
//%% ARCANE_EXPR_SUPPRESS_END
  ExpressionResult(ArrayVariant* data);
  ExpressionResult(IntegerConstArrayView indices);
  ~ExpressionResult();
  
 public:
  void allocate(VariantBase::eType type);

 public:
  ArrayVariant* data() const { return m_data; }
  IntegerConstArrayView indices () const { return m_indices; }
  Integer size() const { return m_indices.size(); }

 public:
  friend std::ostream& operator<<(std::ostream& s, const ExpressionResult& x);

 private:
//%% ARCANE_EXPR_SUPPRESS_BEGIN
  void _init(IVariable* v);
//%% ARCANE_EXPR_SUPPRESS_END

 private:
  ArrayVariant* m_data;
  IntegerConstArrayView m_indices;
  UniqueArray<Integer> m_own_indices; //!< Tableau des indices alloués par cette instance
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

