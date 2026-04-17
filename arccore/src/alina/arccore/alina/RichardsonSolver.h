// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* solver_gmres.h                                              (C) 2000-2026 */
/*                                                                           */
/* Richardson iteration.                                      .              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_RICHARDSONSOLVER_H
#define ARCCORE_ALINA_RICHARDSONSOLVER_H
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
#include "arccore/alina/SolverBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Parameters for Richardson solver.
 */
struct RichardsonSolverParams
{
  using params = RichardsonSolverParams;

  /// Damping factor
  double damping = 1.0;

  /// Maximum number of iterations.
  Int32 maxiter = 100;

  /// Target relative residual error.
  double tol = 1.0e-8;

  /// Target absolute residual error.
  double abstol = std::numeric_limits<double>::min();

  /// Ignore the trivial solution x=0 when rhs is zero.
  //** Useful for searching for the null-space vectors of the system */
  bool ns_search = false;

  /// Verbose output (show iterations and error)
  bool verbose = false;

  RichardsonSolverParams() = default;

  RichardsonSolverParams(const PropertyTree& p)
  : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, damping)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, maxiter)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, tol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, abstol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, ns_search)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, verbose)
  {
    p.check_params({ "damping", "maxiter", "tol", "abstol", "ns_search", "verbose" });
  }

  void get(PropertyTree& p, const std::string& path) const
  {
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, damping);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, maxiter);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, tol);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, abstol);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, ns_search);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, verbose);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Richardson iteration.
 */
template <class Backend, class InnerProduct = detail::default_inner_product>
class RichardsonSolver
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

  using params = RichardsonSolverParams;

 public:

  /// Preallocates necessary data structures for the system of size \p n.
  RichardsonSolver(size_t n,
                   const params& prm = params(),
                   const backend_params& backend_prm = backend_params(),
                   const InnerProduct& inner_product = InnerProduct())
  : prm(prm)
  , n(n)
  , r(Backend::create_vector(n, backend_prm))
  , s(Backend::create_vector(n, backend_prm))
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
  SolverResult operator()(const Matrix& A, const Precond& P, const Vec1& rhs, Vec2&& x) const
  {
    static const coef_type one = math::identity<coef_type>();

    ScopedStreamModifier ss(std::cout);

    scalar_type norm_rhs = norm(rhs);
    if (norm_rhs < Alina::detail::eps<scalar_type>(1)) {
      if (prm.ns_search) {
        norm_rhs = math::identity<scalar_type>();
      }
      else {
        backend::clear(x);
        return SolverResult(0, norm_rhs);
      }
    }

    scalar_type eps = std::max(prm.tol * norm_rhs, prm.abstol);

    backend::residual(rhs, A, x, *r);
    scalar_type res_norm = norm(*r);

    size_t iter = 0;
    for (; iter < prm.maxiter && math::norm(res_norm) > eps; ++iter) {
      P.apply(*r, *s);
      backend::axpby(prm.damping, *s, one, x);
      backend::residual(rhs, A, x, *r);
      res_norm = norm(*r);

      if (prm.verbose && iter % 5 == 0)
        std::cout << iter << "\t" << std::scientific << res_norm / norm_rhs << std::endl;
    }

    return SolverResult(iter, res_norm / norm_rhs);
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
    return backend::bytes(*r) +
    backend::bytes(*s);
  }

  friend std::ostream& operator<<(std::ostream& os, const RichardsonSolver& s)
  {
    return os << "Type:             Richardson"
              << "\nUnknowns:         " << s.n
              << "\nMemory footprint: " << human_readable_memory(s.bytes())
              << std::endl;
  }

 public:

  params prm;

 private:

  size_t n;

  std::shared_ptr<vector> r;
  std::shared_ptr<vector> s;

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
