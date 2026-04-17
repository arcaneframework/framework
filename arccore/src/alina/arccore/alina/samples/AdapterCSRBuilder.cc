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
#include <algorithm>

#include "arccore/alina/AMG.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/ConjugateGradientSolver.h"
#include "arccore/alina/Profiler.h"

#include "SampleProblemCommon.h"

using namespace Arcane;
using namespace Arcane::Alina;

//---------------------------------------------------------------------------
struct poisson_2d
{
  typedef double val_type;
  typedef ptrdiff_t col_type;

  size_t n;
  double h2i;

  poisson_2d(size_t n)
  : n(n)
  , h2i((n - 1) * (n - 1))
  {}

  size_t rows() const { return n * n; }
  size_t nonzeros() const { return 5 * rows(); }

  void operator()(size_t row,
                  std::vector<col_type>& col,
                  std::vector<val_type>& val) const
  {
    size_t i = row % n;
    size_t j = row / n;

    if (j > 0) {
      col.push_back(row - n);
      val.push_back(-h2i);
    }

    if (i > 0) {
      col.push_back(row - 1);
      val.push_back(-h2i);
    }

    col.push_back(row);
    val.push_back(4 * h2i);

    if (i + 1 < n) {
      col.push_back(row + 1);
      val.push_back(-h2i);
    }

    if (j + 1 < n) {
      col.push_back(row + n);
      val.push_back(-h2i);
    }
  }
};

//---------------------------------------------------------------------------

template <class Vec>
double norm(const Vec& v)
{
  return sqrt(Arcane::Alina::backend::inner_product(v, v));
}

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();
  int m = argc > 1 ? atoi(argv[1]) : 1024;
  int n = m * m;

  // Create iterative solver preconditioned by AMG.
  // The use of make_matrix() from crs_builder.hpp allows to construct the
  // system matrix on demand row by row.
  prof.tic("build");
  using Solver = Alina::PreconditionedSolver<Alina::AMG<Alina::BuiltinBackend<double>,
                                                        Alina::SmoothedAggregationCoarserning,
                                                        Alina::GaussSeidelRelaxation>,
                                             Alina::ConjugateGradientSolver<
                                             Alina::BuiltinBackend<double>>>;

  Solver solve(Alina::adapter::make_matrix(poisson_2d(m)));
  prof.toc("build");

  std::cout << solve.precond() << std::endl;

  std::vector<double> f(n, 1);
  std::vector<double> x(n, 0);

  prof.tic("solve");
  SolverResult r = solve(f, x);
  prof.toc("solve");

  std::cout << "Solver:" << std::endl
            << "  Iterations: " << r.nbIteration() << std::endl
            << "  Error:      " << r.residual() << std::endl
            << std::endl;

  // Use the constructed solver as a preconditioner for another iterative
  // solver.
  //
  // Iterative methods use estimated residual for exit condition. For some
  // problems the value of estimated residual can get too far from true
  // residual due to round-off errors.
  //
  // Nesting iterative solvers in this way allows to shave last bits off the
  // error.
  Alina::ConjugateGradientSolver<Alina::BuiltinBackend<double>> S(n);
  std::fill(x.begin(), x.end(), 0);

  prof.tic("nested solver");
  r = S(solve.system_matrix(), solve, f, x);
  prof.toc("nested solver");

  std::cout << "Nested solver:" << std::endl
            << "  Iterations: " << r.nbIteration() << std::endl
            << "  Error:      " << r.residual() << std::endl
            << std::endl;

  std::cout << prof << std::endl;
}
