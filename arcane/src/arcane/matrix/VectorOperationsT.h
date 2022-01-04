// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VectorOperationsT.h                                              (C) 2014 */
/*                                                                           */
/* Arithmetic operators for matrices.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VECTOR_OPERATIONS_H
#define ARCANE_VECTOR_OPERATIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"
//#include "arcane/matrix/VectorExpressionT.h"

#include "VectorExpressionT.h"
#include "IndexedSpace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*!
 * \brief Vector Helper class to handle add and scalar multiply.
 */
template <class V1, class V2>
  class VectorLinComb: public VectorExpr<VectorLinComb<V1,V2> >
{
 public:
    VectorLinComb(double alpha, const VectorExpr<V1>& x,
	       double beta, const VectorExpr<V2>& y);

  ~VectorLinComb() {}

  const IndexedSpace& domain() const;

  Real operator[](int i) const { return 0; }

 private:
  V1 const & m_x;
  V2 const &  m_y;
};


template<class V1, class V2>
  VectorLinComb<V1,V2> operator+(const VectorExpr<V1>& x,
				 const VectorExpr<V2>& y)
{
  return  VectorLinComb<V1,V2>(1, x, 1, y);
}

template<class V1, class V2>
  VectorLinComb<V1,V2> operator-(const VectorExpr<V1>& x,
				 const VectorExpr<V2>& y)
{
  return  VectorLinComb<V1,V2>(1, x, -1, y);
}

template<class V1>
VectorLinComb<V1,V1> operator*(double alpha,
			       const VectorExpr<V1>& x)
{
  return  VectorLinComb<V1,V1>(alpha, x, 0, x);
}


template <class V1, class V2>
  VectorLinComb<V1,V2>::
  VectorLinComb(double alpha, const VectorExpr<V1>& x,
		double beta, const VectorExpr<V2>& y)
  :m_x(x),m_y(y)
{
  // Here: call the correct saxpy when V1 and V2 are not VectorLinComb.
}


template <class V1, class V2>
const IndexedSpace& 
  VectorLinComb<V1,V2>::domain() const {
  return m_x.domain() + m_y.domain();
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_VECTOR_OPERATIONS_H
