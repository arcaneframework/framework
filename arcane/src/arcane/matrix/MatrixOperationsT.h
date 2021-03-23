// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatrixOperationsT.h                                              (C) 2014 */
/*                                                                           */
/* Arithmetic operators for matrices.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATRIX_OPERATIONS_H
#define ARCANE_MATRIX_OPERATIONS_H
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
 * \brief Matrix Helper class to handle add and scalar multiply.
 */
template <class M1, class M2>
  class MatrixLinComb: public MatrixExpr<MatrixLinComb<M1,M2> >
{
 public:
    MatrixLinComb(double alpha, const MatrixExpr<M1>& x,
	       double beta, const MatrixExpr<M2>& y);

  ~MatrixLinComb() {}

  const IndexedSpace& maps_from() const { return m_x.maps_from()+m_y.maps_from(); }
  const IndexedSpace& maps_to() const { return m_x.maps_to()+m_y.maps_to(); }

 private:

  M1 const & m_x;
  M2 const &  m_y;
};


template<class M1, class M2>
  MatrixLinComb<M1,M2> operator+(const MatrixExpr<M1>& x, const MatrixExpr<M2>& y)
{
  return  MatrixLinComb<M1,M2>(1, x, 1, y);
}

template<class M1, class M2>
  MatrixLinComb<M1,M2> operator-(const MatrixExpr<M1>& x, const MatrixExpr<M2>& y)
{
  return  MatrixLinComb<M1,M2>(1, x, -1, y);
}

template<class M1>
MatrixLinComb<M1,M1> operator*(double alpha, const MatrixExpr<M1>& x)
{
  return  MatrixLinComb<M1,M1>(1, x, 0, x);
}


template <class M1, class M2>
  MatrixLinComb<M1,M2>::
  MatrixLinComb(double alpha, const MatrixExpr<M1>& x,
	     double beta, const MatrixExpr<M2>& y)
  :m_x(x),m_y(y)
{
    // Here: call the correct saxpy when M1 and M2 are not MatrixLinComb.
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_MATRIX_OPERATIONS_H
