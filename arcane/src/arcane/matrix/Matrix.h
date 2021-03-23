// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedSpace.h                                                   (C) 2014 */
/*                                                                           */
/* Space for linear algebra.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATRIX_H
#define ARCANE_MATRIX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"
//#include "arcane/matrix/MatrixExpressionT.h"

#include "MatrixExpressionT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*!
 * \brief Matrix class, to be used by user.
 */
class Matrix: public MatrixExpr<Matrix>
{
 public:
 Matrix(const IndexedSpace& from=EmptyIndexedSpace(), const IndexedSpace& to=EmptyIndexedSpace()) 
    : m_from(from), m_to(to) {}

  template <class M>
  Matrix(const MatrixExpr<M>& src) {}

  ~Matrix() {}

  template <class M>
    Matrix& operator=(const MatrixExpr<M>& src) { return *this;}

  const IndexedSpace& maps_from() const { return m_from; }
  const IndexedSpace& maps_to() const { return m_to; }

 private:
  IndexedSpace m_from;
  IndexedSpace m_to;
};



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_MATRIX_H
