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

#include "arccore/alina/CudaBackend.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/BiCGStabSolver.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;
using namespace Arcane::Alina;

int main(int argc, char* argv[])
{
  // The matrix and the RHS file names should be in the command line options:
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <rhs.mtx>" << std::endl;
    return 1;
  }

  // Show the name of the GPU we are using:
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::cout << prop.name << std::endl;

  // The profiler:
  auto& prof = Alina::Profiler::globalProfiler();

  // Read the system matrix and the RHS:
  ptrdiff_t rows, cols;
  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val, rhs;

  prof.tic("read");
  std::tie(rows, cols) = Alina::IO::mm_reader(argv[1])(ptr, col, val);
  std::cout << "Matrix " << argv[1] << ": " << rows << "x" << cols << std::endl;

  std::tie(rows, cols) = Alina::IO::mm_reader(argv[2])(rhs);
  std::cout << "RHS " << argv[2] << ": " << rows << "x" << cols << std::endl;
  prof.toc("read");

  // We use the tuple of CRS arrays to represent the system matrix.
  // Note that std::tie creates a tuple of references, so no data is actually
  // copied here:
  auto A = std::tie(rows, ptr, col, val);

  // Compose the solver type
  using Backend = Alina::backend::cuda<double>;
  typedef PreconditionedSolver<AMG<Backend, SmoothedAggregationCoarserning, SPAI0Relaxation>, BiCGStabSolver<Backend>> Solver;

  // We need to initialize the CUSPARSE library and pass the handle to AMGCL
  // in backend parameters:
  Backend::params bprm;
  cusparseCreate(&bprm.cusparse_handle);

  // There is no way to pass the backend parameters without passing the
  // solver parameters, so we also need to create those. But we can leave
  // them with the default values:
  Solver::params prm;

  // Initialize the solver with the system matrix:
  prof.tic("setup");
  Solver solve(A, prm, bprm);
  prof.toc("setup");

  // Show the mini-report on the constructed solver:
  std::cout << solve << std::endl;

  // Solve the system with the zero initial approximation.
  // The RHS and the solution vectors should reside in the GPU memory:
  thrust::device_vector<double> f(rhs);
  thrust::device_vector<double> x(rows, 0.0);

  prof.tic("solve");
  Alina::SolverResult r = solve(f, x);
  prof.toc("solve");

  // Output the number of iterations, the relative error,
  // and the profiling data:
  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl
            << prof << std::endl;
}
