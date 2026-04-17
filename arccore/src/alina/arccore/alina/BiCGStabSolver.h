// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BiCGStabSolver.h                                            (C) 2000-2026 */
/*                                                                           */
/* BiCGStab iterative method.                                 .              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_BICGSTAB_H
#define ARCCORE_ALINA_BICGSTAB_H
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
 * \brief Parameters for BiConjugate Gradient Stabilized solver.
 */
struct BiCGStabSolverParams
{
  using params = BiCGStabSolverParams;

  /// Preconditioning kind (left/right).
  ePreconditionerSideType pside = ePreconditionerSideType::right;

  /// Maximum number of iterations.
  Int32 maxiter = 100;

  /// Target relative residual error.
  double tol = 1.0e-8;

  /// Target absolute residual error.
  double abstol = std::numeric_limits<double>::min();

  /// Always do at least one iteration.
  bool check_after = false;

  /// Ignore the trivial solution x=0 when rhs is zero.
  //** Useful for searching for the null-space vectors of the system */
  bool ns_search = false;

  /// Verbose output (show iterations and error)
  bool verbose = false;

  BiCGStabSolverParams() = default;

  BiCGStabSolverParams(const PropertyTree& p)
  : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, pside)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, maxiter)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, tol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, abstol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, check_after)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, ns_search)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, verbose)
  {
    p.check_params( { "pside", "maxiter", "tol", "abstol", "check_after", "ns_search", "verbose" });
  }

  void get(PropertyTree& p, const std::string& path) const
  {
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, pside);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, maxiter);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, tol);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, abstol);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, check_after);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, ns_search);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, verbose);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief BiConjugate Gradient Stabilized (BiCGSTAB) method.
 *
 * The BiConjugate Gradient Stabilized method (Bi-CGSTAB) was developed to
 * solve nonsymmetric linear systems while avoiding the often irregular
 * convergence patterns of the Conjugate Gradient [Barr94]_.
 */
template <class Backend, class InnerProduct = detail::default_inner_product>
class BiCGStabSolver
: public SolverBase
{
 public:

  using backend_type = Backend;
  using BackendType = Backend;

  typedef typename Backend::vector vector;
  typedef typename Backend::value_type value_type;
  typedef typename Backend::params backend_params;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  typedef typename math::inner_product_impl<
  typename math::rhs_of<value_type>::type>::return_type coef_type;

  using params = BiCGStabSolverParams;

  /// Preallocates necessary data structures for the system of size \p n.
  BiCGStabSolver(size_t n,
                 const params& prm = params(),
                 const backend_params& backend_prm = backend_params(),
                 const InnerProduct& inner_product = InnerProduct())
  : prm(prm)
  , n(n)
  , r(Backend::create_vector(n, backend_prm))
  , p(Backend::create_vector(n, backend_prm))
  , v(Backend::create_vector(n, backend_prm))
  , s(Backend::create_vector(n, backend_prm))
  , t(Backend::create_vector(n, backend_prm))
  , rh(Backend::create_vector(n, backend_prm))
  , T(Backend::create_vector(n, backend_prm))
  , inner_product(inner_product)
  {}

  /* Computes the solution for the given system matrix \p A and the
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
    static const coef_type zero = math::zero<coef_type>();

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

    if (prm.pside == ePreconditionerSideType::left) {
      backend::residual(rhs, A, x, *rh);
      P.apply(*rh, *r);
    }
    else {
      backend::residual(rhs, A, x, *r);
    }
    backend::copy(*r, *rh);

    scalar_type eps = std::max(norm_rhs * prm.tol, prm.abstol);
    scalar_type res = prm.check_after ? 2 * eps : norm(*r);

    coef_type rho1 = zero;
    coef_type rho2 = zero;
    coef_type alpha = zero;
    coef_type omega = zero;

    Int32 iter = 0;
    for (bool first = true; res > eps && iter < prm.maxiter; ++iter) {

      rho2 = rho1;
      rho1 = inner_product(*r, *rh);

      if (first) {
        backend::copy(*r, *p);
        first = false;
      }
      else {
        precondition(!math::is_zero(rho2), "Zero rho in BiCGStab");
        coef_type beta = (rho1 * alpha) / (rho2 * omega);
        backend::axpbypcz(one, *r, -beta * omega, *v, beta, *p);
      }

      preconditioner_spmv(prm.pside, P, A, *p, *v, *T);

      alpha = rho1 / inner_product(*rh, *v);

      if (prm.pside == ePreconditionerSideType::left) {
        backend::axpby(alpha, *p, one, x);
      }
      else {
        backend::axpby(alpha, *T, one, x);
      }

      backend::axpbypcz(one, *r, -alpha, *v, zero, *s);

      if ((res = norm(*s)) > eps) {
        preconditioner_spmv(prm.pside, P, A, *s, *t, *T);

        omega = inner_product(*t, *s) / inner_product(*t, *t);

        precondition(!math::is_zero(omega), "Zero omega in BiCGStab");

        if (prm.pside == ePreconditionerSideType::left) {
          backend::axpby(omega, *s, one, x);
        }
        else {
          backend::axpby(omega, *T, one, x);
        }

        backend::axpbypcz(one, *s, -omega, *t, zero, *r);

        res = norm(*r);
      }

      if (prm.verbose && iter % 5 == 0)
        std::cout << iter << "\t" << std::scientific << res / norm_rhs << std::endl;
    }

    return SolverResult(iter, res / norm_rhs);
  }

  /*
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
    backend::bytes(*p) +
    backend::bytes(*v) +
    backend::bytes(*s) +
    backend::bytes(*t) +
    backend::bytes(*rh) +
    backend::bytes(*T);
  }

  friend std::ostream& operator<<(std::ostream& os, const BiCGStabSolver& s)
  {
    return os << "Type:             BiCGStab"
              << "\nUnknowns:         " << s.n
              << "\nMemory footprint: " << human_readable_memory(s.bytes())
              << std::endl;
  }

 public:

  params prm;

 private:

  size_t n;

  std::shared_ptr<vector> r;
  std::shared_ptr<vector> p;
  std::shared_ptr<vector> v;
  std::shared_ptr<vector> s;
  std::shared_ptr<vector> t;
  std::shared_ptr<vector> rh;
  std::shared_ptr<vector> T;

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
