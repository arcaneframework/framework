// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* solver_gmres.h                                              (C) 2000-2026 */
/*                                                                           */
/* GMRES solver method.                                       .              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_GMRESSOLVER_H
#define ARCCORE_ALINA_GMRESSOLVER_H
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
 * \brief Parameters for GMRES solver.
 */
struct GMRESSolverParams
{
  using params = GMRESSolverParams;

  /// Number of iterations before restart.
  Int32 M = 30;

  /// Preconditioning kind (left/right).
  ePreconditionerSideType pside = ePreconditionerSideType::right;

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

  GMRESSolverParams() = default;

  GMRESSolverParams(const PropertyTree& p)
  : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, M)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, pside)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, maxiter)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, tol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, abstol)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, ns_search)
  , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, verbose)
  {
    p.check_params({ "M", "pside", "maxiter", "tol", "abstol", "ns_search", "verbose" });
  }

  void get(PropertyTree& p, const std::string& path) const
  {
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, M);
    ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, pside);
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
 * \brief Generalized Minimal Residual (GMRES) method.
 *
 * \rst
 * The Generalized Minimal Residual method is an extension of MINRES (which is
 * only applicable to symmetric systems) to unsymmetric systems [Barr94]_.
 * \endrst
 */
template <class Backend, class InnerProduct = detail::default_inner_product>
class GMRESSolver
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
  typedef typename math::inner_product_impl<rhs_type>::return_type coef_type;

  using params = GMRESSolverParams;

  /// Preallocates necessary data structures for the system of size \p n.
  GMRESSolver(size_t n,
              const params& prm = params(),
              const backend_params& backend_prm = backend_params(),
              const InnerProduct& inner_product = InnerProduct())
  : prm(prm)
  , n(n)
  , H(prm.M + 1, prm.M)
  , s(prm.M + 1)
  , cs(prm.M + 1)
  , sn(prm.M + 1)
  , r(Backend::create_vector(n, backend_prm))
  , inner_product(inner_product)
  {
    v.reserve(prm.M + 1);
    for (unsigned i = 0; i <= prm.M; ++i)
      v.push_back(Backend::create_vector(n, backend_prm));
  }

  /*!
   * Computes the solution for the given system matrix.
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
  SolverResult operator()(Matrix const& A, Precond const& P, Vec1 const& rhs, Vec2& x) const
  {
    static const scalar_type zero = math::zero<scalar_type>();
    static const scalar_type one = math::identity<scalar_type>();

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
    scalar_type norm_r = zero;

    size_t iter = 0;
    while (true) {
      if (prm.pside == ePreconditionerSideType::left) {
        backend::residual(rhs, A, x, *v[0]);
        P.apply(*v[0], *r);
      }
      else {
        backend::residual(rhs, A, x, *r);
      }

      // -- Check stopping condition
      norm_r = norm(*r);
      if (norm_r < eps || iter >= prm.maxiter)
        break;

      // -- Inner GMRES iteration
      backend::axpby(math::inverse(norm_r), *r, zero, *v[0]);

      std::fill(s.begin(), s.end(), 0);
      s[0] = norm_r;

      unsigned j = 0;
      while (true) {
        // -- Arnoldi process
        //
        // Build an orthonormal basis V and matrix H such that
        //     A V_{i-1} = V_{i} H
        vector& v_new = *v[j + 1];

        preconditioner_spmv(prm.pside, P, A, *v[j], v_new, *r);

        for (unsigned k = 0; k <= j; ++k) {
          H(k, j) = inner_product(v_new, *v[k]);
          backend::axpby(-H(k, j), *v[k], one, v_new);
        }
        H(j + 1, j) = norm(v_new);

        backend::axpby(math::inverse(H(j + 1, j)), v_new, zero, v_new);

        for (unsigned k = 0; k < j; ++k)
          detail::apply_plane_rotation(H(k, j), H(k + 1, j), cs[k], sn[k]);

        detail::generate_plane_rotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        detail::apply_plane_rotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        detail::apply_plane_rotation(s[j], s[j + 1], cs[j], sn[j]);

        scalar_type inner_res = std::abs(s[j + 1]);

        if (prm.verbose && iter % 5 == 0)
          std::cout << iter << "\t" << std::scientific << inner_res / norm_rhs << std::endl;

        // Check for termination
        ++j, ++iter;
        if (iter >= prm.maxiter || j >= prm.M || inner_res <= eps)
          break;
      }

      // -- GMRES terminated: eval solution
      for (unsigned i = j; i-- > 0;) {
        s[i] /= H(i, i);
        for (unsigned k = 0; k < i; ++k)
          s[k] -= H(k, i) * s[i];
      }

      // -- Apply step
      vector& dx = *r;
      backend::lin_comb(j, s, v, zero, dx);

      if (prm.pside == ePreconditionerSideType::left) {
        backend::axpby(one, dx, one, x);
      }
      else {
        vector& tmp = *v[0];
        P.apply(dx, tmp);
        backend::axpby(one, tmp, one, x);
      }
    }

    return std::make_tuple(iter, norm_r / norm_rhs);
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
  std::tuple<size_t, scalar_type>
  operator()(Precond const& P, Vec1 const& rhs, Vec2& x) const
  {
    return (*this)(P.system_matrix(), P, rhs, x);
  }

  friend std::ostream& operator<<(std::ostream& os, const GMRESSolver& s)
  {
    return os << "Type:             GMRES(" << s.prm.M << ")"
              << "\nUnknowns:         " << s.n
              << "\nMemory footprint: " << human_readable_memory(s.bytes())
              << std::endl;
  }

 public:

  params prm;

  size_t bytes() const
  {
    size_t b = 0;

    b += H.size() * sizeof(coef_type);
    b += backend::bytes(s);
    b += backend::bytes(cs);
    b += backend::bytes(sn);
    b += backend::bytes(*r);

    for (const auto& x : v)
      b += backend::bytes(*x);

    return b;
  }

 private:

  size_t n;

  mutable multi_array<coef_type, 2> H;
  mutable std::vector<coef_type> s, cs, sn;
  std::shared_ptr<vector> r;
  std::vector<std::shared_ptr<vector>> v;

  InnerProduct inner_product;

  template <class Vec>
  scalar_type norm(const Vec& x) const
  {
    return std::abs(sqrt(inner_product(x, x)));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
