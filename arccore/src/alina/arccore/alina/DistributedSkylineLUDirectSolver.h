// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedSkylineLUDirectSolver.h                          (C) 2000-2026 */
/*                                                                           */
/* Distributed direct solver that uses Skyline LU factorization.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDSKYLINELUDIRECTSOLVER_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDSKYLINELUDIRECTSOLVER_H
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
#include "arccore/alina/Adapters.h"
#include "arccore/alina/SkylineLUSolver.h"
#include "arccore/alina/DistributedDirectSolverBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Provides distributed direct solver interface for Skyline LU solver.
 *
 * This is a wrapper around Skyline LU factorization solver that provides a
 * distributed direct solver interface but always works sequentially.
 */
template <typename Backend>
class DistributedSkylineLUDirectSolver
: public DistributedDirectSolverBase<Backend, DistributedSkylineLUDirectSolver<Backend>>
{
  using Base = DistributedDirectSolverBase<Backend, DistributedSkylineLUDirectSolver<Backend>>;

 public:

  typedef typename Backend::value_type value_type;
  typedef Alina::solver::SkylineLUSolver<value_type> Solver;
  typedef typename Solver::params params;
  typedef Backend::matrix build_matrix;

  /// Constructor.
  template <class Matrix>
  DistributedSkylineLUDirectSolver(mpi_communicator comm, const Matrix& A,
                                   const params& prm = params{})
  : prm(prm)
  {
    static_cast<Base*>(this)->init(comm, A);
  }

  static size_t coarse_enough()
  {
    return Solver::coarse_enough();
  }

  int comm_size(int /*n*/) const
  {
    return 1;
  }

  void init(mpi_communicator, const build_matrix& A)
  {
    S = std::make_shared<Solver>(A, prm);
  }

  /*!
   * \brief Solves the problem for the given right-hand side.
   *
   * \param rhs The right-hand side.
   * \param x   The solution.
   */
  template <class Vec1, class Vec2>
  void solve(const Vec1& rhs, Vec2& x) const
  {
    (*S)(rhs, x);
  }

 private:

  params prm;
  std::shared_ptr<Solver> S;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
