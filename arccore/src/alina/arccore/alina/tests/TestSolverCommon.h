// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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

#ifndef TESTS_TEST_SOLVER_HPP
#define TESTS_TEST_SOLVER_HPP

#include "arccore/alina/AMG.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/Profiler.h"

#include <boost/assign/std/vector.hpp>
using namespace boost::assign;

#include "SampleProblemCommon.h"

using namespace Arcane;

namespace
{
Arcane::Alina::Profiler prof;
}

//---------------------------------------------------------------------------

template <class Backend, class Matrix>
void test_solver(const Matrix& A,
                 std::shared_ptr<typename Backend::vector> const& f,
                 std::shared_ptr<typename Backend::vector>& x,
                 Alina::eSolverType solver,
                 Alina::eRelaxationType relaxation,
                 Alina::eCoarserningType coarsening,
                 typename Backend::params const& bprm,
                 bool test_null_space = false)
{
  Alina::PropertyTree prm;
  prm.put("precond.coarse_enough", 500);
  prm.put("precond.coarsening.type", coarsening);
  prm.put("precond.relax.type", relaxation);
  prm.put("solver.type", solver);

  typedef typename Backend::value_type value_type;
  std::vector<double> null;

  if (test_null_space && Alina::math::static_rows<value_type>::value == 1) {
    Int64 n = Alina::backend::nbRow(*A);
    null.resize(n, 1.0);

    prm.put("precond.coarsening.nullspace.cols", 1);
    prm.put("precond.coarsening.nullspace.rows", n);
    prm.put("precond.coarsening.nullspace.B", &null[0]);
  }

  Alina::PreconditionedSolver<Alina::AMG<Backend, Alina::CoarseningRuntime, Alina::RelaxationRuntime>,
                              Alina::SolverRuntime<Backend>>
  solve(A, prm, bprm);

  std::cout << solve.precond() << std::endl;

  Alina::backend::clear(*x);

  Alina::SolverResult r = solve(*f, *x);

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl
            << std::endl;

  ASSERT_NEAR(r.residual(), 0.0, 1e-4);
}

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
void test_rap(const Matrix& A,
              std::shared_ptr<typename Backend::vector> const& f,
              std::shared_ptr<typename Backend::vector>& x,
              Alina::eSolverType solver,
              Alina::eRelaxationType relaxation,
              typename Backend::params const& bprm)
{
  Alina::PropertyTree prm;
  prm.put("precond.type", relaxation);
  prm.put("solver.type", solver);

  Alina::PreconditionedSolver<Alina::RelaxationAsPreconditioner<Backend, Alina::RelaxationRuntime>,
                              Alina::SolverRuntime<Backend>>
  solve(A, prm, bprm);

  std::cout << "Using " << relaxation << " as preconditioner" << std::endl;

  Alina::backend::clear(*x);

  Alina::SolverResult r = solve(*f, *x);

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl;

  ASSERT_NEAR(r.residual(), 0.0, 1e-4);
}

template <class Backend, class value_type, class col_type, class ptr_type, class rhs_type>
void test_problem(size_t n,
                  std::vector<ptr_type> ptr,
                  std::vector<col_type> col,
                  std::vector<value_type> val,
                  std::vector<rhs_type> rhs,
                  typename Backend::params const& bprm)
{
  Alina::eCoarserningType coarsening[] = {
    Alina::eCoarserningType::aggregation,
    Alina::eCoarserningType::smoothed_aggregation,
    Alina::eCoarserningType::smoothed_aggr_emin,
    Alina::eCoarserningType::ruge_stuben
  };

  Alina::eRelaxationType relaxation[] = {
    Alina::eRelaxationType::spai0,
    Alina::eRelaxationType::spai1,
    Alina::eRelaxationType::damped_jacobi,
    Alina::eRelaxationType::gauss_seidel,
    Alina::eRelaxationType::ilu0,
    Alina::eRelaxationType::iluk,
    Alina::eRelaxationType::ilup,
    Alina::eRelaxationType::ilut,
    Alina::eRelaxationType::chebyshev
  };

  Alina::eSolverType solver[] = {
    Alina::eSolverType::cg,
    Alina::eSolverType::bicgstab,
    Alina::eSolverType::bicgstabl,
    Alina::eSolverType::gmres,
    Alina::eSolverType::lgmres,
    Alina::eSolverType::fgmres,
    Alina::eSolverType::idrs
  };

  typename Backend::params prm;

  auto y = Backend::copy_vector(rhs, prm);
  auto x = Backend::create_vector(n, prm);

  // Test solvers
  for (Alina::eSolverType s : solver) {
    std::cout << "Solver: " << s << std::endl;
    try {
      test_solver<Backend>(Alina::adapter::zero_copy_direct(n, ptr.data(), col.data(), val.data()),
                           y, x, s, relaxation[0], coarsening[0], bprm);
    }
    catch (const std::logic_error&) {
    }
  }

  // Test smoothers
  for (Alina::eRelaxationType r : relaxation) {
    std::cout << "Relaxation: " << r << std::endl;
    try {
      test_solver<Backend>(Alina::adapter::zero_copy_direct(n, ptr.data(), col.data(), val.data()),
                           y, x, solver[0], r, coarsening[0], bprm);
    }
    catch (const std::logic_error&) {
    }

    try {
      std::cout << "Relaxation as preconditioner: " << r << std::endl;

      test_rap<Backend>(Alina::adapter::zero_copy_direct(n, ptr.data(), col.data(), val.data()),
                        y, x, solver[0], r, bprm);
    }
    catch (const std::logic_error&) {
    }
  }

  // Test coarsening
  for (Alina::eCoarserningType c : coarsening) {
    std::cout << "Coarsening: " << c << std::endl;

    try {
      test_solver<Backend>(Alina::adapter::zero_copy_direct(n, ptr.data(), col.data(), val.data()),
                           y, x, solver[0], relaxation[0], c, bprm);
    }
    catch (const std::logic_error&) {
    }

    switch (c) {
    case Alina::eCoarserningType::aggregation:
    case Alina::eCoarserningType::smoothed_aggregation:
    case Alina::eCoarserningType::smoothed_aggr_emin:
      test_solver<Backend>(Alina::adapter::zero_copy_direct(n, ptr.data(), col.data(), val.data()),
                           y, x, solver[0], relaxation[0], c, bprm, /*test_null_space*/ true);
      break;
    default:
      break;
    }
  }
}

template <class Backend>
void test_backend(typename Backend::params const& bprm = typename Backend::params())
{
  typedef typename Backend::value_type value_type;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;
  typedef typename Alina::math::rhs_of<value_type>::type rhs_type;

  // Poisson 3D
  {
    std::vector<ptr_type> ptr;
    std::vector<col_type> col;
    std::vector<value_type> val;
    std::vector<rhs_type> rhs;

    size_t n = sample_problem(24, val, col, ptr, rhs);

    test_problem<Backend>(n, ptr, col, val, rhs, bprm);
  }

  // Trivial problem
  {
    std::vector<ptr_type> ptr;
    std::vector<col_type> col;
    std::vector<value_type> val;
    std::vector<rhs_type> rhs;

    val += Alina::math::identity<value_type>(), Alina::math::identity<value_type>();
    col += 0, 1;
    ptr += 0, 1, 2;
    rhs += Alina::math::constant<rhs_type>(1.0), Alina::math::zero<rhs_type>();

    size_t n = rhs.size();

    test_problem<Backend>(n, ptr, col, val, rhs, bprm);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_HAS_ACCELERATOR_THREAD)
#define ARCCORE_ALINA_TEST_DO_TEST_TASK(name1, name2, func) \
  TEST(name1, name2##_task4) \
  { \
    func(false, 4); \
  }
#else
#define ARCCORE_ALINA_TEST_DO_TEST_TASK(name1, name2, func)
#endif

#define ARCCORE_ALINA_TEST_DO_TEST_SEQUENTIAL(name1, name2, func) \
  TEST(name1, name2) \
  { \
    func(false, 0); \
  }

//! Macro pour définir les tests en fonction de l'accélérateur
#define ARCCORE_ALINA_TEST_DO_TEST_ACCELERATOR(name1, name2, func) \
  ARCCORE_ALINA_TEST_DO_TEST_TASK(name1, name2, func); \
  ARCCORE_ALINA_TEST_DO_TEST_SEQUENTIAL(name1, name2, func);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
