// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* solver_preonly.h                                            (C) 2000-2026 */
/*                                                                           */
/* Solver which only apply preconditioner once.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PRECONDITIONERONLYSOLVER_H
#define ARCCORE_ALINA_PRECONDITIONERONLYSOLVER_H
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

#include "arccore/alina/SolverUtils.h"
#include "arccore/alina/AlinaUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Solver which only apply preconditioner once.
 */
template <class Backend, class InnerProduct = detail::default_inner_product>
class PreconditionerOnlySolver
: public SolverBase
{
 public:

  typedef Backend backend_type;

  typedef typename Backend::vector vector;
  typedef typename Backend::value_type value_type;
  typedef typename Backend::params backend_params;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  typedef typename math::inner_product_impl<
  typename math::rhs_of<value_type>::type>::return_type coef_type;

  /// Solver parameters.
  typedef Alina::detail::empty_params params;

  /// Preallocates necessary data structures for the system of size \p n.
  PreconditionerOnlySolver(size_t n,
                           const params& = params(),
                           const backend_params& = backend_params(),
                           const InnerProduct& inner_product = InnerProduct())
  : n(n)
  , inner_product(inner_product)
  {}

  /*!
   * \brief Computes the solution for the given system matrix.
   *
   * Computes the solution for the given system matrix \p A and the
   * right-hand side \p rhs.  Returns the number of iterations made and
   * the achieved residual as a ``std::tuple``. The solution vector
   * \p x provides initial approximation in input and holds the computed
   * solution on output.
   *
   * The system matrix may differ from the matrix used during
   * initialization. This may be used for the solution of non-stationary
   * problems with slowly changing coefficients. There is a strong chance
   * that a preconditioner built for a time step will act as a reasonably
   * good preconditioner for several subsequent time steps [DeSh12]_.
   */
  template <class Matrix, class Precond, class Vec1, class Vec2>
  SolverResult operator()(const Matrix&, const Precond& P, const Vec1& rhs, Vec2&& x) const
  {
    P.apply(rhs, x);
    return SolverResult{};
  }

  /*!
   * \brief Computes the solution for the given right-hand side.
   *
   * Computes the solution for the given right-hand side \p rhs. The
   * system matrix is the same that was used for the setup of the
   * preconditioner \p P.  Returns the number of iterations made and the
   * achieved residual as a ``std::tuple``. The solution vector \p x
   * provides initial approximation in input and holds the computed
   * solution on output.
   */
  template <class Precond, class Vec1, class Vec2>
  SolverResult operator()(const Precond& P, const Vec1& rhs, Vec2&& x) const
  {
    return (*this)(P.system_matrix(), P, rhs, x);
  }

  size_t bytes() const
  {
    return 0;
  }

  friend std::ostream& operator<<(std::ostream& os, const PreconditionerOnlySolver& s)
  {
    return os
    << "Type:             PreOnly"
    << "\nUnknowns:         " << s.n
    << "\nMemory footprint: " << human_readable_memory(s.bytes())
    << std::endl;
  }

 private:

  size_t n;

  InnerProduct inner_product;

  template <class Vec>
  scalar_type norm(const Vec& x) const
  {
    return sqrt(math::norm(inner_product(x, x)));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
