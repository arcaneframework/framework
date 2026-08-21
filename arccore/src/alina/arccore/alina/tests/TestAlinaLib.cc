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

#include "arccore/alina/AlinaLib.h"
#include "./SampleProblemCommon.h"

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

TEST(alina_test_alina_lib, basic)
{
  std::cout << "Testing AlinaLib\n";

  std::vector<int> ptr;
  std::vector<int> col;
  std::vector<double> val;
  std::vector<double> rhs;

  int n = sample_problem(12l, val, col, ptr, rhs);

  AlinaParameters prm;

  prm.setInt32("precond.coarse_enough", 1000);
  prm.setString("precond.coarsening.type", "smoothed_aggregation");
  prm.setReal("precond.coarsening.aggr.eps_strong", 1e-3f);
  prm.setString("precond.relax.type", "spai0");

  prm.setString("solver.type", "bicgstabl");
  prm.setInt32("solver.L", 1);
  prm.setInt32("solver.maxiter", 100);

  AlinaCSRMatrixView matrix_view(n, ptr.data(), col.data(), val.data());

  AlinaSequentialSolver solver(matrix_view, &prm);

  std::vector<double> x(n, 0);
  AlinaConvergenceInfo cnv = solver.solve(rhs.data(), x.data());

  // Solve same problem again, but explicitly provide the matrix this time:
  std::fill(x.begin(), x.end(), 0);
  cnv = solver.solveMatrix(matrix_view, rhs.data(), x.data());

  std::cout << "Iterations: " << cnv.iterations << std::endl
            << "Error:      " << cnv.residual << std::endl;
}
