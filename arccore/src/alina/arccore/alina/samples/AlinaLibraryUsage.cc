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

#include <iostream>
#include <vector>

#include <../AlinaLib.h>
#include "SampleProblemCommon.h"

int main()
{
  std::vector<int> ptr;
  std::vector<int> col;
  std::vector<double> val;
  std::vector<double> rhs;

  int n = sample_problem(12l, val, col, ptr, rhs);

  AlinaParameters* prm = AlinaLib::params_create();

  AlinaLib::params_set_int(prm, "precond.coarse_enough", 1000);
  AlinaLib::params_set_string(prm, "precond.coarsening.type", "smoothed_aggregation");
  AlinaLib::params_set_float(prm, "precond.coarsening.aggr.eps_strong", 1e-3f);
  AlinaLib::params_set_string(prm, "precond.relax.type", "spai0");

  AlinaLib::params_set_string(prm, "solver.type", "bicgstabl");
  AlinaLib::params_set_int(prm, "solver.L", 1);
  AlinaLib::params_set_int(prm, "solver.maxiter", 100);

  AlinaSequentialSolver* solver = AlinaLib::solver_create(n, ptr.data(), col.data(), val.data(), prm);

  AlinaLib::params_destroy(prm);

  std::vector<double> x(n, 0);
  AlinaConvergenceInfo cnv = AlinaLib::solver_solve(solver, rhs.data(), x.data());

  // Solve same problem again, but explicitly provide the matrix this time:
  std::fill(x.begin(), x.end(), 0);
  cnv = AlinaLib::solver_solve_matrix(
  solver, ptr.data(), col.data(), val.data(),
  rhs.data(), x.data());

  std::cout << "Iterations: " << cnv.iterations << std::endl
            << "Error:      " << cnv.residual << std::endl;

  AlinaLib::solver_destroy(solver);
}
