// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeflatedSolver.h                                            (C) 2000-2026 */
/*                                                                           */
/* Iterative preconditioned solver with deflation.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_DEFLATEDSOLVER_H
#define ARCCORE_ALINA_DEFLATEDSOLVER_H
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

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/DenseMatrixInverseImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Iterative preconditioned solver with deflation.
 */
template <class Precond, class IterativeSolver>
class DeflatedSolver
: public detail::non_copyable
{
  static_assert(backend::backends_compatible<
                typename IterativeSolver::backend_type,
                typename Precond::backend_type>::value,
                "Backends for preconditioner and iterative solver should be compatible");

 public:

  typedef typename IterativeSolver::backend_type backend_type;
  typedef typename backend_type::matrix matrix;
  typedef typename backend_type::vector vector;

  typedef typename backend_type::value_type value_type;
  typedef typename backend_type::params backend_params;
  typedef typename BuiltinBackend<value_type>::matrix build_matrix;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  /*!
   * \brief Combined parameters of the bundled preconditioner
   * and the iterative solver.
   */
  struct params
  {
    int nvec = 0; ///< The number of deflation vectors
    scalar_type* vec = nullptr; ///< Deflation vectors as a [nvec x n] matrix

    typename Precond::params precond; ///< Preconditioner parameters.
    typename IterativeSolver::params solver; ///< Iterative solver parameters.

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, nvec)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, vec)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, precond)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, solver)
    {
      p.check_params({ "nvec", "vec", "precond", "solver" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, nvec);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, vec);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, precond);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, solver);
    }
  };

  /** Sets up the preconditioner and creates the iterative solver. */
  template <class Matrix>
  DeflatedSolver(const Matrix& A,
                 const params& prm = params(),
                 const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(A))
  , P(A, prm.precond, bprm)
  , S(backend::nbRow(A), prm.solver, bprm)
  , r(backend_type::create_vector(n, bprm))
  , Z(prm.nvec)
  , E(prm.nvec * prm.nvec, 0)
  , d(prm.nvec)
  {
    init(A, bprm);
  }

  // Constructs the preconditioner and creates iterative solver.
  // Takes shared pointer to the matrix in internal format.
  DeflatedSolver(std::shared_ptr<build_matrix> A,
                 const params& prm = params(),
                 const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(*A))
  , P(A, prm.precond, bprm)
  , S(backend::nbRow(*A), prm.solver, bprm)
  , r(backend_type::create_vector(n, bprm))
  , Z(prm.nvec)
  , E(prm.nvec * prm.nvec, 0)
  , d(prm.nvec)
  {
    init(*A, bprm);
  }

  template <class Matrix>
  void init(const Matrix& A, const backend_params& bprm)
  {
    precondition(prm.nvec > 0 && prm.vec != nullptr, "Deflation vectors are not set!");

    for (int i = 0; i < prm.nvec; ++i) {
      SmallSpan<scalar_type> irange(prm.vec + n * i, n);
      Z[i] = backend_type::copy_vector(std::make_shared<numa_vector<scalar_type>>(irange), bprm);
    }

    std::vector<scalar_type> AZ(prm.nvec);
    std::fill(E.begin(), E.end(), math::zero<scalar_type>());
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      std::fill(AZ.begin(), AZ.end(), math::zero<scalar_type>());
      for (auto a = backend::row_begin(A, i); a; ++a) {
        for (int j = 0; j < prm.nvec; ++j) {
          AZ[j] += a.value() * prm.vec[j * n + a.col()];
        }
      }

      for (int ii = 0, k = 0; ii < prm.nvec; ++ii) {
        for (int jj = 0; jj < prm.nvec; ++jj, ++k) {
          E[k] += prm.vec[i + ii * n] * AZ[jj];
        }
      }
    }

    std::vector<scalar_type> t(E.size());
    std::vector<int> p(prm.nvec);
    detail::inverse(prm.nvec, E.data(), t.data(), p.data());
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
   * \rst
   * The system matrix may differ from the matrix used during
   * initialization. This may be used for the solution of non-stationary
   * problems with slowly changing coefficients. There is a strong chance
   * that a preconditioner built for a time step will act as a reasonably
   * good preconditioner for several subsequent time steps [DeSh12]_.
   * \endrst
   */
  template <class Matrix, class Vec1, class Vec2>
  SolverResult operator()(const Matrix& A, const Vec1& rhs, Vec2&& x) const
  {
    project(rhs, x);
    return S(A, *this, rhs, x);
  }

  /*!
   * \brief Computes the solution for the given right-hand.
   *
   * Computes the solution for the given right-hand side \p rhs.
   * Returns the number of iterations made and the achieved residual as a
   * ``std::tuple``. The solution vector \p x provides initial
   * approximation in input and holds the computed solution on output.
   */
  template <class Vec1, class Vec2>
  SolverResult operator()(const Vec1& rhs, Vec2&& x) const
  {
    project(rhs, x);
    return S(*this, rhs, x);
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    P.apply(rhs, x);
    project(rhs, x);
  }

  template <class Vec1, class Vec2>
  void project(const Vec1& b, Vec2& x) const
  {
    // x += Z^T E^{-1} Z (b - Ax)
    backend::residual(b, P.system_matrix(), x, *r);
    std::fill(d.begin(), d.end(), math::zero<scalar_type>());
    for (int j = 0; j < prm.nvec; ++j) {
      auto fj = backend::inner_product(*Z[j], *r);
      for (int i = 0; i < prm.nvec; ++i)
        d[i] += E[i * prm.nvec + j] * fj;
    }
    backend::lin_comb(prm.nvec, d, Z, 1, x);
  }

  /// Returns reference to the constructed preconditioner.
  const Precond& precond() const
  {
    return P;
  }

  /// Returns reference to the constructed preconditioner.
  Precond& precond()
  {
    return P;
  }

  /// Returns reference to the constructed iterative solver.
  const IterativeSolver& solver() const
  {
    return S;
  }

  /// Returns the system matrix in the backend format.
  std::shared_ptr<typename Precond::matrix> system_matrix_ptr() const
  {
    return P.system_matrix_ptr();
  }

  typename Precond::matrix const& system_matrix() const
  {
    return P.system_matrix();
  }

  /// Stores the parameters used during construction into the property tree \p p.
  void get_params(Alina::PropertyTree& p) const
  {
    prm.get(p);
  }

  /// Returns the size of the system matrix.
  size_t size() const
  {
    return n;
  }

  size_t bytes() const
  {
    return backend::bytes(S) + backend::bytes(P);
  }

  friend std::ostream& operator<<(std::ostream& os, const DeflatedSolver& p)
  {
    return os << "Solver\n======\n"
              << p.S << std::endl
              << "Preconditioner\n==============\n"
              << p.P;
  }

 public:

  params prm;

 private:

  size_t n;
  Precond P;
  IterativeSolver S;
  std::shared_ptr<vector> r;
  std::vector<std::shared_ptr<vector>> Z;
  std::vector<scalar_type> E;
  mutable std::vector<scalar_type> d;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
