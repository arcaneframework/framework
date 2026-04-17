// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedPreconditionedSolver.h                           (C) 2000-2026 */
/*                                                                           */
/* Iterative solver wrapper for distributed linear systems.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_DISTRIBUTEDPRECONDITIONEDSOLVER_H
#define ARCCORE_ALINA_DISTRIBUTEDPRECONDITIONEDSOLVER_H
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

#include <iostream>

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/DistributedInnerProduct.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Iterative solver wrapper for distributed linear systems.
 */
template <class Precond, class IterativeSolver>
class DistributedPreconditionedSolver
: public Alina::detail::non_copyable
{
  static_assert(backend::backends_compatible<
                typename IterativeSolver::BackendType,
                typename Precond::BackendType>::value,
                "Backends for preconditioner and iterative solver should be compatible");

 public:

  typedef typename IterativeSolver::BackendType backend_type;
  using BackendType = backend_type;
  typedef DistributedMatrix<typename Precond::BackendType> matrix;
  typedef typename backend_type::value_type value_type;
  typedef typename backend_type::params backend_params;
  typedef typename BuiltinBackend<value_type>::matrix build_matrix;
  typedef typename math::scalar_of<value_type>::type scalar_type;

  struct params
  {
    typename Precond::params precond; ///< Preconditioner parameters.
    typename IterativeSolver::params solver; ///< Iterative solver parameters.

    params() {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, precond)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, solver)
    {
      p.check_params({ "precond", "solver" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, precond);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, solver);
    }
  } prm;

  template <class Matrix>
  DistributedPreconditionedSolver(mpi_communicator comm, const Matrix& A,
                                  const params& prm = params(),
                                  const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(A))
  , P(comm, A, prm.precond, bprm)
  , S(backend::nbRow(A), prm.solver, bprm, DistributedInnerProduct(comm))
  {}

  DistributedPreconditionedSolver(mpi_communicator comm,
                                  std::shared_ptr<matrix> A,
                                  const params& prm = params(),
                                  const backend_params& bprm = backend_params())
  : prm(prm)
  , n(A->loc_rows())
  , P(comm, A, prm.precond, bprm)
  , S(n, prm.solver, bprm, DistributedInnerProduct(comm))
  {
  }

  template <class Backend>
  DistributedPreconditionedSolver(mpi_communicator comm,
                                  std::shared_ptr<DistributedMatrix<Backend>> A,
                                  const params& prm = params(),
                                  const backend_params& bprm = backend_params())
  : prm(prm)
  , n(A->loc_rows())
  , P(comm, std::make_shared<matrix>(*A), prm.precond, bprm)
  , S(n, prm.solver, bprm, DistributedInnerProduct(comm))
  {
    A->move_to_backend(bprm);
  }

  DistributedPreconditionedSolver(mpi_communicator comm, std::shared_ptr<build_matrix> A,
                                  const params& prm = params(),
                                  const backend_params& bprm = backend_params())
  : prm(prm)
  , n(backend::nbRow(*A))
  , P(comm, A, prm.precond, bprm)
  , S(backend::nbRow(*A), prm.solver, bprm, DistributedInnerProduct(comm))
  {}

  template <class Matrix, class Vec1, class Vec2>
  SolverResult operator()(const Matrix& A, const Vec1& rhs, Vec2&& x) const
  {
    return S(A, P, rhs, x);
  }

  template <class Vec1, class Vec2>
  SolverResult operator()(const Vec1& rhs, Vec2&& x) const
  {
    return S(P, rhs, x);
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    backend::clear(x);
    (*this)(rhs, x);
  }

  const Precond& precond() const
  {
    return P;
  }

  Precond& precond()
  {
    return P;
  }

  const IterativeSolver& solver() const
  {
    return S;
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return P.system_matrix_ptr();
  }

  const matrix& system_matrix() const
  {
    return P.system_matrix();
  }

  void get_params(Alina::PropertyTree& p) const
  {
    prm.get(p);
  }

  size_t size() const
  {
    return n;
  }

  friend std::ostream& operator<<(std::ostream& os, const DistributedPreconditionedSolver& M)
  {
    return os << M.S << std::endl
              << M.P;
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
