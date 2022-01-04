// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatrixVectorOperationsT.h                                        (C) 2014 */
/*                                                                           */
/* Arithmetic operators for matrices and vectors.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATRIX_VECTOR_OPERATIONS_H
#define ARCANE_MATRIX_VECTOR_OPERATIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"
//#include "arcane/matrix/MatrixExpressionT.h"

#include "MatrixExpressionT.h"
#include "VectorExpressionT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*!
 * \brief Matrix Helper class to handle add and scalar multiply.
 */
template <class M, class V>
  class VectorAXPY: public VectorExpr<VectorAXPY<M,V> >
{
 public:
    VectorAXPY(const MatrixExpr<M>& a,
	       const VectorExpr<V>& y);

  ~VectorAXPY() {}

  const IndexedSpace& domain() const;

 private:
  M const & m_a;
  V const & m_x;
};


template<class M, class V>
  VectorAXPY<M,V> operator+(const MatrixExpr<M>& a, const VectorExpr<V>& x)
{
  return  VectorAXPY<M,V>(a,x);
}

template <class M, class V>
  VectorAXPY<M,V>::VectorAXPY(const MatrixExpr<M>& a,
			      const VectorExpr<V>& y)
  :m_a(a),m_x(x)
{
    
}

template <class M, class V>
const IndexedSpace&
VectorAXPY<M,V>::domain() const ()
{
  if (!m_a.maps_from().isCompatible(m_x.domain())) {
    // Throw an exception.
  }

  return m_a.maps_to();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_MATRIX_VECTOR_OPERATIONS_H
