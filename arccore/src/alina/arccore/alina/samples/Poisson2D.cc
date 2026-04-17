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

#include "arccore/alina/AMG.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/BiCGStabSolver.h"

#include <iostream>
#include <vector>
#include <tuple>

using namespace Arcane;

struct poisson_2d
{
  typedef double val_type;
  typedef long col_type;

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

int main(int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();

  int m = argc > 1 ? atoi(argv[1]) : 1024;
  int n = m * m;

  using Backend = Alina::BuiltinBackend<double>;
  std::vector<double> f(n);
  std::vector<double> x(n);

  prof.tic("build");

  using Precond = Alina::AMG<Backend, Alina::SmoothedAggregationCoarserning, Alina::SPAI0Relaxation>;
  using Solver = Alina::PreconditionedSolver<Precond, Alina::BiCGStabSolver<Backend>>;

  Solver solve(Alina::adapter::make_matrix(poisson_2d(m)));
  prof.toc("build");

  //std::cout << solve.amg() << std::endl;

  std::fill_n(f.data(), n, 1.0);
  std::fill_n(x.data(), n, 0.0);

  prof.tic("solve");
  Alina::SolverResult r = solve(f, x);
  prof.toc("solve");

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl
            << std::endl;

  std::cout << prof << std::endl;
}
