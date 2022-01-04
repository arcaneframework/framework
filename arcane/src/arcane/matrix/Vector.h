// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedSpace.h                                                   (C) 2014 */
/*                                                                           */
/* Space for linear algebra.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VECTOR_H
#define ARCANE_VECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"
//#include "arcane/matrix/MatrixExpressionT.h"

#include "VectorExpressionT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*!
 * \brief Vector class, to be used by user.
 */
class Vector: public VectorExpr<Vector>
{
 public:
  Vector(const IndexedSpace& domain): m_domain(domain) {}
  template <class V> Vector(const VectorExpr<V>& src) {}

  ~Vector() {}

  template <class V>
    Vector& operator=(const VectorExpr<V>& src) { return *this;}

  Real operator[](int i) const { return 0; }

  const IndexedSpace& domain() const { return m_domain; }

 private:
  IndexedSpace m_domain;
};



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_VECTOR_H
