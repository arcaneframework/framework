// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDDRSolver.h                                                (C) 2000-2026 */
/*                                                                           */
/* IDR(s) (Induced Dimension Reduction) method.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_IDRSSOLVER_H
#define ARCCORE_ALINA_IDRSSOLVER_H
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
/*
 * The code is ported from Matlab code published at
 * http://ta.twi.tudelft.nl/nw/users/gijzen/IDR.html.
 *
 * This is a very stable and efficient IDR(s) variant (implemented in the MATLAB
 * code idrs.m given above) as described in: Martin B. van Gijzen and Peter
 * Sonneveld, Algorithm 913: An Elegant IDR(s) Variant that Efficiently Exploits
 * Bi-orthogonality Properties. ACM Transactions on Mathematical Software, Vol.
 * 38, No. 1, pp. 5:1-5:19, 2011 (copyright ACM).
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/SolverUtils.h"
#include "arccore/alina/AlinaUtils.h"

#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Parameters for IDR(s) solver.
 */
struct IDRSSolverParams
{
  using params = IDRSSolverParams;

  /// Dimension of the shadow space in IDR(s).
  Int32 s = 4;

  /*!
   * \brief Computation of omega.
   *
   * If omega = 0: a standard minimum residual step is performed
   * If omega > 0: omega is increased if
   * the cosine of the angle between Ar and r < omega
   * Default: omega = 0.7;
   */
  double omega = 0.7;

  /// Specifies if residual smoothing must be applied.
  bool smoothing = false;

  /*!
   * \brief Residual replacement.
   *
   * Determines the residual replacement strategy.
   * If true, the recursively computed residual is replaced by the
   * true residual.
   * Default: No residual replacement.
   */
  bool replacement = false;

  /// Maximum number of iterations.
  Int32 maxiter = 100;

  /// Target relative residual error.
  double tol = 1e-8;

  /// Target absolute residual error.
  double abstol = std::numeric_limits<double>::min();

  /*!
   * \brief Ignore the trivial solution x=0 when rhs is zero.
   *
   * Useful for searching for the null-space vectors of the system.
   */
  bool ns_search = false;

  /// Verbose output (show iterations and error)
  bool verbose = false;

  IDRSSolverParams() = default;

  IDRSSolverParams(const PropertyTree& p)
  : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, s)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, omega)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, smoothing)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, replacement)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, maxiter)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, tol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, abstol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, ns_search)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, verbose)
  {
    p.check_params({ "s", "omega", "smoothing", "replacement", "maxiter", "tol", "abstol", "ns_search", "verbose" });
  }

  void get(PropertyTree& p, const std::string& path) const
  {
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, s);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, omega);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, smoothing);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, replacement);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, maxiter);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, tol);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, abstol);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, ns_search);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, verbose);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// IDR(s) method (Induced Dimension Reduction)
template <class Backend, class InnerProduct = detail::default_inner_product>
class IDRSSolver
: public SolverBase
{
 public:

  using backend_type = Backend;
  using BackendType = Backend;

  typedef typename Backend::vector vector;
  typedef typename Backend::value_type value_type;
  typedef typename Backend::params backend_params;

  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef typename math::rhs_of<value_type>::type rhs_type;

  typedef typename math::inner_product_impl<typename math::rhs_of<value_type>::type>::return_type coef_type;

  using params = IDRSSolverParams;

  /// Preallocates necessary data structures for the system of size \p n.
  IDRSSolver(size_t n,
             const params& prm = params(),
             const backend_params& bprm = backend_params(),
             const InnerProduct& inner_product = InnerProduct())
  : prm(prm)
  , n(n)
  , inner_product(inner_product)
  , M(prm.s, prm.s)
  , f(prm.s)
  , c(prm.s)
  , r(Backend::create_vector(n, bprm))
  , v(Backend::create_vector(n, bprm))
  , t(Backend::create_vector(n, bprm))
  {
    static const scalar_type one = math::identity<scalar_type>();
    static const scalar_type zero = math::zero<scalar_type>();

    if (prm.smoothing) {
      x_s = Backend::create_vector(n, bprm);
      r_s = Backend::create_vector(n, bprm);
    }

    G.reserve(prm.s);
    U.reserve(prm.s);
    for (Int32 i = 0; i < prm.s; ++i) {
      G.push_back(Backend::create_vector(n, bprm));
      U.push_back(Backend::create_vector(n, bprm));
    }

    // Initialize P.
    P.reserve(prm.s);
    {
      std::vector<rhs_type> p(n);

      int pid = inner_product.rank();
      const int nt = 1;

      const int tid = 0;
      std::mt19937 rng(pid * nt + tid);
      std::uniform_real_distribution<scalar_type> rnd(-1, 1);

      for (unsigned j = 0; j < prm.s; ++j) {
        for (ptrdiff_t i = 0; i < n; ++i) {
          p[i] = math::constant<rhs_type>(rnd(rng));
        }
        P.push_back(Backend::copy_vector(p, bprm));
      }

      for (unsigned j = 0; j < prm.s; ++j) {
        for (unsigned k = 0; k < j; ++k) {
          coef_type alpha = inner_product(*P[k], *P[j]);
          backend::axpby(-alpha, *P[k], one, *P[j]);
        }
        scalar_type norm_pj = norm(*P[j]);
        backend::axpby(math::inverse(norm_pj), *P[j], zero, *P[j]);
      }
    }
  }

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
  SolverResult operator()(Matrix const& A, Precond const& Prec, Vec1 const& rhs, Vec2& x) const
  {
    static const scalar_type one = math::identity<scalar_type>();
    static const scalar_type zero = math::zero<scalar_type>();

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

    // Compute initial residual:
    backend::residual(rhs, A, x, *r);

    scalar_type res_norm = norm(*r);
    if (res_norm <= eps) {
      // Initial guess is a good enough solution.
      return SolverResult(0, res_norm / norm_rhs);
    }

    if (prm.smoothing) {
      backend::copy(x, *x_s);
      backend::copy(*r, *r_s);
    }

    // Initialization.
    coef_type om = math::identity<coef_type>();

    for (unsigned i = 0; i < prm.s; ++i) {
      backend::clear(*G[i]);
      backend::clear(*U[i]);

      for (unsigned j = 0; j < prm.s; ++j)
        M(i, j) = (i == j);
    }

    // Main iteration loop, build G-spaces:
    size_t iter = 0;
    while (iter < prm.maxiter && res_norm > eps) {
      // New righ-hand size for small system:
      for (unsigned i = 0; i < prm.s; ++i)
        f[i] = inner_product(*r, *P[i]);

      for (unsigned k = 0; k < prm.s; ++k) {
        // Compute new v
        backend::copy(*r, *v);

        // Solve small system (Note: M is lower triangular)
        // and make v orthogonal to P:
        for (unsigned i = k; i < prm.s; ++i) {
          c[i] = f[i];
          for (unsigned j = k; j < i; ++j)
            c[i] -= M(i, j) * c[j];
          c[i] = math::inverse(M(i, i)) * c[i];

          backend::axpby(-c[i], *G[i], one, *v);
        }

        Prec.apply(*v, *t);

        // Compute new U[k]
        backend::axpby(om, *t, c[k], *U[k]);
        for (unsigned i = k + 1; i < prm.s; ++i)
          backend::axpby(c[i], *U[i], one, *U[k]);

        // Compute new G[k], G[k] is in space G_j
        backend::spmv(one, A, *U[k], zero, *G[k]);

        // Bi-Orthogonalise the new basis vectors:
        for (unsigned i = 0; i < k; ++i) {
          coef_type alpha = inner_product(*G[k], *P[i]) / M(i, i);

          backend::axpby(-alpha, *G[i], one, *G[k]);
          backend::axpby(-alpha, *U[i], one, *U[k]);
        }

        // New column of M = P'*G  (first k-1 entries are zero)
        for (unsigned i = k; i < prm.s; ++i)
          M(i, k) = inner_product(*G[k], *P[i]);

        precondition(!math::is_zero(M(k, k)), "IDR(s) breakdown: zero M[k,k]");

        // Make r orthogonal to q_i, i = [0..k)
        coef_type beta = math::inverse(M(k, k)) * f[k];
        backend::axpby(-beta, *G[k], one, *r);
        backend::axpby(beta, *U[k], one, x);

        res_norm = norm(*r);

        // Smoothing
        if (prm.smoothing) {
          backend::axpbypcz(one, *r_s, -one, *r, zero, *t);
          coef_type gamma = inner_product(*t, *r_s) / inner_product(*t, *t);
          backend::axpby(-gamma, *t, one, *r_s);
          backend::axpbypcz(-gamma, *x_s, gamma, x, one, *x_s);
          res_norm = norm(*r_s);
        }

        if (prm.verbose && iter % 5 == 0)
          std::cout << iter << "\t" << std::scientific << res_norm / norm_rhs << std::endl;
        if (res_norm <= eps || ++iter >= prm.maxiter)
          break;

        // New f = P'*r (first k  components are zero)
        for (unsigned i = k + 1; i < prm.s; ++i)
          f[i] -= beta * M(i, k);
      }

      if (res_norm <= eps || iter >= prm.maxiter)
        break;

      // Now we have sufficient vectors in G_j to compute residual in G_j+1
      // Note: r is already perpendicular to P so v = r

      Prec.apply(*r, *v);
      backend::spmv(one, A, *v, zero, *t);

      // Computation of a new omega
      om = omega(*t, *r);
      precondition(!math::is_zero(om), "IDR(s) breakdown: zero omega");

      backend::axpby(-om, *t, one, *r);
      backend::axpby(om, *v, one, x);

      if (prm.replacement) {
        backend::residual(rhs, A, x, *r);
      }
      res_norm = norm(*r);

      // Smoothing.
      if (prm.smoothing) {
        backend::axpbypcz(one, *r_s, -one, *r, zero, *t);
        coef_type gamma = inner_product(*t, *r_s) / inner_product(*t, *t);
        backend::axpby(-gamma, *t, one, *r_s);
        backend::axpbypcz(-gamma, *x_s, gamma, x, one, *x_s);
        res_norm = norm(*r_s);
      }

      ++iter;
    }

    if (prm.smoothing)
      backend::copy(*x_s, x);

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
  SolverResult operator()(Precond const& P, Vec1 const& rhs, Vec2& x) const
  {
    return (*this)(P.system_matrix(), P, rhs, x);
  }

  size_t bytes() const
  {
    size_t b = 0;

    b += M.size() * sizeof(coef_type);

    b += backend::bytes(f);
    b += backend::bytes(c);

    b += backend::bytes(*r);
    b += backend::bytes(*v);
    b += backend::bytes(*t);

    if (x_s)
      b += backend::bytes(*x_s);
    if (r_s)
      b += backend::bytes(*r_s);

    for (const auto& v : P)
      b += backend::bytes(*v);
    for (const auto& v : G)
      b += backend::bytes(*v);
    for (const auto& v : U)
      b += backend::bytes(*v);

    return b;
  }

  friend std::ostream& operator<<(std::ostream& os, const IDRSSolver& s)
  {
    return os << "Type:             IDR(" << s.prm.s << ")"
              << "\nUnknowns:         " << s.n
              << "\nMemory footprint: " << human_readable_memory(s.bytes())
              << std::endl;
  }

 public:

  params prm;

 private:

  size_t n;

  InnerProduct inner_product;

  mutable multi_array<coef_type, 2> M;
  mutable std::vector<coef_type> f, c;

  std::shared_ptr<vector> r, v, t;
  std::shared_ptr<vector> x_s;
  std::shared_ptr<vector> r_s;

  std::vector<std::shared_ptr<vector>> P, G, U;

  template <class Vec>
  scalar_type norm(const Vec& x) const
  {
    return std::abs(sqrt(inner_product(x, x)));
  }

  template <class Vector1, class Vector2>
  coef_type omega(const Vector1& t, const Vector2& s) const
  {
    scalar_type norm_t = norm(t);
    scalar_type norm_s = norm(s);

    coef_type ts = inner_product(t, s);
    scalar_type rho = math::norm(ts / (norm_t * norm_s));
    coef_type om = ts / (norm_t * norm_t);

    if (rho < prm.omega)
      om *= prm.omega / rho;

    return om;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
