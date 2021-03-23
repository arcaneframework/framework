// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatrixExpressionT.h                                              (C) 2014 */
/*                                                                           */
/* Virtual Matrix to perform arithmetic operations.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATRIX_EXPRESSION_H
#define ARCANE_MATRIX_EXPRESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"
//#include "arcane/matrix/IndexedSpace.h"

#include "IndexedSpace.h"
#include "fake.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*!
 * \brief Class to inherite to perform matrix computations.
 */

template <class M>
class MatrixExpr
{
 public:

  operator M&()             { return *static_cast<      M*>(this); }
  operator M const&() const { return *static_cast<const M*>(this); }

  const IndexedSpace& maps_from() const 
  { return static_cast<M const&>(*this).maps_from(); }

  const IndexedSpace& maps_to() const 
  { return static_cast<M const&>(*this).maps_to(); }


 private:
  
};



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_MATRIX_EXPRESSION_H
