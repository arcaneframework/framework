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

#include <vector>
#include <tuple>

#include "arccore/alina/Adapters.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/ConjugateGradientSolver.h"
#include "arccore/alina/Profiler.h"

#ifndef SOLVER_BACKEND_BUILTIN
#define SOLVER_BACKEND_BUILTIN
#endif
#include "arccore/alina/BuiltinBackend.h"
typedef Arcane::Alina::BuiltinBackend<float> fBackend;
typedef Arcane::Alina::BuiltinBackend<double> dBackend;

#include "SampleProblemCommon.h"

using namespace Arcane;
using namespace Arcane::Alina;

int main()
{
  // Combine single-precision preconditioner with a
  // double-precision Krylov solver.
  typedef Alina::PreconditionedSolver<Alina::AMG<fBackend,
                                        Alina::SmoothedAggregationCoarserning,
                                        Alina::SPAI0Relaxation>,
                             Alina::ConjugateGradientSolver<dBackend>>
  Solver;

  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val, rhs;

  dBackend::params bprm;

  ARCCORE_ALINA_TIC("assemble");
  int n = sample_problem(128, val, col, ptr, rhs);
  ARCCORE_ALINA_TOC("assemble");

  auto A_d = std::tie(n, ptr, col, val);
  std::vector<double>& f = rhs;
  std::vector<double> x(n, 0.0);

  ARCCORE_ALINA_TIC("setup");
  Solver S(std::tie(n, ptr, col, val), Solver::params(), bprm);
  ARCCORE_ALINA_TIC("setup");

  std::cout << S << std::endl;

  ARCCORE_ALINA_TIC("solve");
  SolverResult r = S(A_d, f, x);
  ARCCORE_ALINA_TIC("solve");

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl
            << Alina::Profiler::globalProfiler() << std::endl;
}
