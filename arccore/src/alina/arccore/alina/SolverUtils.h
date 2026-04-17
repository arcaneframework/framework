// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SolverUtils.h                                               (C) 2000-2026 */
/*                                                                           */
/* Utility functions for solver classes.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_SOLVERUTILS_H
#define ARCCORE_ALINA_SOLVERUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/ValueTypeInterface.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class ePreconditionerSideType
{
  left,
  right
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream& operator<<(std::ostream& os, ePreconditionerSideType p)
{
  switch (p) {
  case ePreconditionerSideType::left:
    return os << "left";
  case ePreconditionerSideType::right:
    return os << "right";
  default:
    return os << "???";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::istream& operator>>(std::istream& in, ePreconditionerSideType& p)
{
  std::string val;
  in >> val;

  if (val == "left")
    p = ePreconditionerSideType::left;
  else if (val == "right")
    p = ePreconditionerSideType::right;
  else
    throw std::invalid_argument("Invalid preconditioning side. "
                                "Valid choices are: left, right.");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Preconditioned matrix-vector product
template <class Precond, class Matrix, class VecF, class VecX, class VecT> void
preconditioner_spmv(ePreconditionerSideType pside, const Precond& P, const Matrix& A,
                    const VecF& F, VecX& X, VecT& T)
{
  typedef typename backend::value_type<Matrix>::type value;
  typedef typename math::scalar_of<value>::type scalar;

  static const scalar one = math::identity<scalar>();
  static const scalar zero = math::zero<scalar>();

  if (pside == ePreconditionerSideType::left) {
    backend::spmv(one, A, F, zero, T);
    P.apply(T, X);
  }
  else {
    P.apply(F, T);
    backend::spmv(one, A, T, zero, X);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::detail
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Default inner product getter for iterative solvers.
 *
 * Falls through to backend::inner_product().
 */
struct default_inner_product
{
  template <class Vec1, class Vec2>
  typename math::inner_product_impl<typename backend::value_type<Vec1>::type>::return_type
  operator()(const Vec1& x, const Vec2& y) const
  {
    return backend::inner_product(x, y);
  }

  int rank() const
  {
    return 0;
  }
};

//! Givens plane rotations used in GMRES variants.
template <class T>
void generate_plane_rotation(T dx, T dy, T& cs, T& sn)
{
  if (math::is_zero(dy)) {
    cs = 1;
    sn = 0;
  }
  else if (std::abs(dy) > std::abs(dx)) {
    T tmp = dx / dy;
    sn = math::inverse(sqrt(math::identity<T>() + tmp * tmp));
    cs = tmp * sn;
  }
  else {
    T tmp = dy / dx;
    cs = math::inverse(sqrt(math::identity<T>() + tmp * tmp));
    sn = tmp * cs;
  }
}

template <class T>
void apply_plane_rotation(T& dx, T& dy, T cs, T sn)
{
  T tmp = math::adjoint(cs) * dx + math::adjoint(sn) * dy;
  dy = -sn * dx + cs * dy;
  dx = tmp;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
