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

#include <vector>
#include <iostream>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/BiCGStabSolver.h"
#include "arccore/alina/StaticMatrix.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;

int main(int argc, char* argv[])
{
  // The command line should contain the matrix file name:
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx>" << std::endl;
    return 1;
  }

  // The profiler:
  auto& prof = Alina::Profiler::globalProfiler();

  // Read the system matrix:
  ptrdiff_t rows, cols;
  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val;

  prof.tic("read");
  std::tie(rows, cols) = Alina::IO::mm_reader(argv[1])(ptr, col, val);
  std::cout << "Matrix " << argv[1] << ": " << rows << "x" << cols << std::endl;
  prof.toc("read");

  // The RHS is filled with ones:
  std::vector<double> f(rows, 1.0);

  // Scale the matrix so that it has the unit diagonal.
  // First, find the diagonal values:
  std::vector<double> D(rows, 1.0);
  for (ptrdiff_t i = 0; i < rows; ++i) {
    for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
      if (col[j] == i) {
        D[i] = 1 / sqrt(val[j]);
        break;
      }
    }
  }

  // Then, apply the scaling in-place:
  for (ptrdiff_t i = 0; i < rows; ++i) {
    for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
      val[j] *= D[i] * D[col[j]];
    }
    f[i] *= D[i];
  }

  // We use the tuple of CRS arrays to represent the system matrix.
  // Note that std::tie creates a tuple of references, so no data is actually
  // copied here:
  auto A = std::tie(rows, ptr, col, val);

  // Compose the solver type
  typedef Alina::StaticMatrix<double, 4, 4> dmat_type; // matrix value type in double precision
  typedef Alina::StaticMatrix<double, 4, 1> dvec_type; // the corresponding vector value type
  typedef Alina::StaticMatrix<float, 4, 4> smat_type; // matrix value type in single precision

  typedef Alina::BuiltinBackend<dmat_type> SBackend; // the solver backend
  typedef Alina::BuiltinBackend<smat_type> PBackend; // the preconditioner backend

  using Preconditioner = Alina::AMG<PBackend,
                                    Alina::SmoothedAggregationCoarserning,
                                    Alina::ILU0Relaxation>;
  using Solver = Alina::PreconditionedSolver<Preconditioner, Alina::BiCGStabSolver<SBackend>>;

  // Initialize the solver with the system matrix.
  // Use the block_matrix adapter to convert the matrix into
  // the block format on the fly:
  prof.tic("setup");
  auto Ab = Alina::adapter::block_matrix<dmat_type>(A);
  Solver solve(Ab);
  prof.toc("setup");

  // Show the mini-report on the constructed solver:
  std::cout << solve << std::endl;

  // Solve the system with the zero initial approximation:
  std::vector<double> x(rows, 0.0);

  // Reinterpret both the RHS and the solution vectors as block-valued:
  auto f_ptr = reinterpret_cast<dvec_type*>(f.data());
  auto x_ptr = reinterpret_cast<dvec_type*>(x.data());
  auto F = SmallSpan<dvec_type>(f_ptr, rows / 4);
  auto X = SmallSpan<dvec_type>(x_ptr, rows / 4);

  prof.tic("solve");
  Alina::SolverResult r = solve(Ab, F, X);
  prof.toc("solve");

  // Output the number of iterations, the relative error,
  // and the profiling data:
  std::cout << "Iters: " << r.nbIteration() << std::endl
            << "Error: " << r.residual() << std::endl
            << prof << std::endl;
}
