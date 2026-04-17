// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PreconditionedSolver.h                                      (C) 2000-2026 */
/*                                                                           */
/* Tie an iterative solver and a preconditioner in a single class.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PRECONDITIONEDSOLVER_H
#define ARCCORE_ALINA_PRECONDITIONEDSOLVER_H
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Convenience class that bundles together a preconditioner and an iterative solver.
 */
template <class Precond, class IterativeSolver>
class PreconditionedSolver
: public Alina::detail::non_copyable
{
  static_assert(backend::backends_compatible<
                typename IterativeSolver::backend_type,
                typename Precond::backend_type>::value,
                "Backends for preconditioner and iterative solver should be compatible");

 public:

  typedef typename IterativeSolver::backend_type backend_type;
  typedef typename backend_type::matrix matrix;

  typedef typename backend_type::value_type value_type;
  typedef typename backend_type::col_type col_type;
  typedef typename backend_type::ptr_type ptr_type;
  typedef typename backend_type::params backend_params;
  typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  /*!
   * \brief Combined parameters of the bundled preconditioner and the iterative solver.
   */
  struct params
  {
    typename Precond::params precond; ///< Preconditioner parameters.
    typename IterativeSolver::params solver; ///< Iterative solver parameters.

    params() {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, precond)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, solver)
    {
      p.check_params( { "precond", "solver" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, precond);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, solver);
    }
  } prm;

  /*!
   * \brief Sets up the preconditioner and creates the iterative solver.
   */
  template <class Matrix>
  PreconditionedSolver(const Matrix& A,
                       const params& prm = params(),
                       const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(A))
  , P(A, prm.precond, bprm)
  , S(backend::nbRow(A), prm.solver, bprm)
  {}

  // Constructs the preconditioner and creates iterative solver.
  // Takes shared pointer to the matrix in internal format.
  PreconditionedSolver(std::shared_ptr<build_matrix> A,
                       const params& prm = params(),
                       const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(*A))
  , P(A, prm.precond, bprm)
  , S(backend::nbRow(*A), prm.solver, bprm)
  {}

  /** Computes the solution for the given system matrix \p A and the
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
    return S(A, P, rhs, x);
  }

  /** Computes the solution for the given right-hand side \p rhs.
   * Returns the number of iterations made and the achieved residual as a
   * ``std::tuple``. The solution vector \p x provides initial
   * approximation in input and holds the computed solution on output.
   */
  template <class Vec1, class Vec2>
  SolverResult operator()(const Vec1& rhs, Vec2&& x) const
  {
    return S(P, rhs, x);
  }

  /*!
   * Acts as a preconditioner. That is, applies the solver to the
   * right-hand side \p rhs to get the solution \p x with zero initial
   * approximation.  Iterative methods usually use estimated residual for
   * exit condition.  For some problems the value of the estimated
   * residual can get too far from the true residual due to round-off
   * errors.  Nesting iterative solvers in this way may allow to shave
   * the last bits off the error. The method should not be used directly
   * but rather allows nesting ``make_solver`` classes as in the
   * following example:
   *
   * \rst
   * .. code-block:: cpp
   *
   *   typedef ::Arcane::Alina::PreconditionedSolver<
   *     ::Arcane::Alina::PreconditionedSolver<
   *       ::Arcane::Alina::AMG<
   *         Backend, ::Arcane::Alina::coarsening::smoothed_aggregation, ::Arcane::Alina::relaxation::spai0
   *         >,
   *       ::Arcane::Alina::solver::cg<Backend>
   *       >,
   *     ::Arcane::Alina::solver::cg<Backend>
   *     > NestedSolver;
   * \endrst
   */
  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    backend::clear(x);
    (*this)(rhs, x);
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

  /// Returns reference to the constructed iterative solver.
  IterativeSolver& solver()
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

  friend std::ostream& operator<<(std::ostream& os, const PreconditionedSolver& p)
  {
    return os << "Solver\n======\n"
              << p.S << std::endl
              << "Preconditioner\n==============\n"
              << p.P;
  }

 private:

  size_t n;
  Precond P;
  IterativeSolver S;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
