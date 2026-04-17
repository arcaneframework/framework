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

#include <gtest/gtest.h>

// To remove warnings about deprecated Eigen usage.
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <Eigen/SparseLU>
#include "arccore/alina/EigenSolver.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/Profiler.h"
#include "SampleProblemCommon.h"

namespace
{
Arcane::Alina::Profiler prof;
}

using namespace Arcane;

TEST(alina_test_solvers, eigen_solver)
{
  std::vector<int> ptr;
  std::vector<int> col;
  std::vector<double> val;
  std::vector<double> rhs;

  size_t n = sample_problem(16, val, col, ptr, rhs);
  Alina::CSRMatrix<double> A(std::tie(n, ptr, col, val));

  using Solver = Alina::EigenSolver<Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>>;

  Solver solve(A);

  std::vector<double> x(n);
  std::vector<double> r(n);

  solve(rhs, x);

  Alina::backend::residual(rhs, A, x, r);

  ASSERT_NEAR(sqrt(Alina::backend::inner_product(r, r)), 0.0, 1e-8);
}
