// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/*
The MIT License

Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <string>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/SchurPressureCorrectionPreconditioner.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/make_block_solver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/ConjugateGradientSolver.h"
#include "arccore/alina/PreconditionerOnlySolver.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/Relaxation.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // The command line should contain the matrix and the RHS file names,
  // and the number of unknowns in the flow subsytem:
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <matrix.bin> <rhs.bin> <nu>" << std::endl;
    return 1;
  }

  // The profiler:
  auto& prof = Alina::Profiler::globalProfiler();

  // Read the system matrix:
  ptrdiff_t rows, cols;
  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val, rhs;

  prof.tic("read");
  Alina::IO::read_crs(argv[1], rows, ptr, col, val);
  Alina::IO::read_dense(argv[2], rows, cols, rhs);
  std::cout << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl;
  std::cout << "RHS " << argv[2] << ": " << rows << "x" << cols << std::endl;
  prof.toc("read");

  // The number of unknowns in the U subsystem
  ptrdiff_t nu = std::stoi(argv[3]);

  // We use the tuple of CRS arrays to represent the system matrix.
  // Note that std::tie creates a tuple of references, so no data is actually
  // copied here:
  auto A = std::tie(rows, ptr, col, val);

  // Compose the solver type
  typedef Alina::BuiltinBackend<double> SBackend; // the outer iterative solver backend
  typedef Alina::BuiltinBackend<float> PBackend; // the PSolver backend
  typedef Alina::BuiltinBackend<Alina::StaticMatrix<float, 3, 3>> UBackend; // the USolver backend

  typedef Alina::PreconditionedSolver<
  Alina::preconditioner::SchurPressureCorrectionPreconditioner<
  Alina::make_block_solver<Alina::AMG<
                           UBackend,
                           Alina::AggregationCoarsening,
                           Alina::ILU0Relaxation>,
                           Alina::PreconditionerOnlySolver<UBackend>>,
  Alina::PreconditionedSolver<Alina::RelaxationAsPreconditioner<
                              PBackend,
                              Alina::SPAI0Relaxation>,
                              Alina::PreconditionerOnlySolver<PBackend>>>,
  Alina::ConjugateGradientSolver<SBackend>>
  Solver;

  // Solver parameters
  Solver::params prm;
  prm.precond.simplec_dia = false;
  prm.precond.pmask.resize(rows);
  for (ptrdiff_t i = 0; i < rows; ++i)
    prm.precond.pmask[i] = (i >= nu);

  // Initialize the solver with the system matrix.
  prof.tic("setup");
  Solver solve(A, prm);
  prof.toc("setup");

  // Show the mini-report on the constructed solver:
  std::cout << solve << std::endl;

  // Solve the system with the zero initial approximation:
  std::vector<double> x(rows, 0.0);
  prof.tic("solve");
  Alina::SolverResult r = solve(A, rhs, x);
  prof.toc("solve");

  // Output the number of iterations, the relative error,
  // and the profiling data:
  std::cout << "Iters: " << r.nbIteration() << std::endl
            << "Error: " << r.residual() << std::endl
            << prof << std::endl;
}
