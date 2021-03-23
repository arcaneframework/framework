// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VectorExpressionT.h                                              (C) 2014 */
/*                                                                           */
/* Virtual Vector to perform arithmetic operations.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VECTOR_EXPRESSION_H
#define ARCANE_VECTOR_EXPRESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"

#include "fake.h"
#include "IndexedSpace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*!
 * \brief Class to inherite to perform matrix computations.
 */

template <class V>
class VectorExpr
{
 public:

  operator V&()             { return *static_cast<      V*>(this); }
  operator V const&() const { return *static_cast<const V*>(this); }
  /* ICC does not work with : */
  /* operator V&() { return static_cast<V&>(*this); } */

  Real operator[](int i) const {
    return static_cast<V const&>(*this)[i];     }
  
  const IndexedSpace& domain() const 
  { return static_cast<V const&>(*this).domain(); }

 private:
};



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_VECTOR_EXPRESSION_H
