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

#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/DistributedPreconditionedSolver.h"
#include "arccore/alina/DistributedAMG.h"
#include "arccore/alina/DistributedCoarsening.h"
#include "arccore/alina/DistributedRelaxation.h"
#include "arccore/alina/DistributedSolver.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
#include "arccore/alina/ParmetisMatrixPartitioner.h"
#endif

using namespace Arcane;
using namespace Arcane::Alina;

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // The matrix and the RHS file names should be in the command line options:
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix.bin> <rhs.bin>" << std::endl;
    return 1;
  }

  Alina::mpi_init mpi(&argc, &argv);
  Alina::mpi_communicator world(MPI_COMM_WORLD);

  auto& prof = Alina::Profiler::globalProfiler();

  // Read the system matrix and the RHS:
  prof.tic("read");
  // Get the global size of the matrix:
  ptrdiff_t rows = Alina::IO::crs_size<ptrdiff_t>(argv[1]);
  ptrdiff_t cols;

  // Split the matrix into approximately equal chunks of rows
  ptrdiff_t chunk = (rows + world.size - 1) / world.size;
  ptrdiff_t row_beg = std::min(rows, chunk * world.rank);
  ptrdiff_t row_end = std::min(rows, row_beg + chunk);
  chunk = row_end - row_beg;

  // Read our part of the system matrix and the RHS.
  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val, rhs;
  Alina::IO::read_crs(argv[1], rows, ptr, col, val, row_beg, row_end);
  Alina::IO::read_dense(argv[2], rows, cols, rhs, row_beg, row_end);
  prof.toc("read");

  if (world.rank == 0)
    std::cout
    << "World size: " << world.size << std::endl
    << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl
    << "RHS " << argv[2] << ": " << rows << "x" << cols << std::endl;

  // Compose the solver type
  typedef BuiltinBackend<double> DBackend;
  typedef BuiltinBackend<float> FBackend;
  using Solver = DistributedPreconditionedSolver<DistributedAMG<
                                                 FBackend,
                                                 DistributedSmoothedAggregationCoarsening<FBackend>,
                                                 DistributedSPAI0Relaxation<FBackend>>,
                                                 DistributedBiCGStabSolver<DBackend>>;

  // Create the distributed matrix from the local parts.
  auto A = std::make_shared<DistributedMatrix<DBackend>>(world, std::tie(chunk, ptr, col, val));

  // Partition the matrix and the RHS vector.
  // If neither ParMETIS not PT-SCOTCH are not available,
  // just keep the current naive partitioning.
#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
  typedef Alina::ParmetisMatrixPartitioner<DBackend> Partition;

  if (world.size > 1) {
    prof.tic("partition");
    Partition part;

    // part(A) returns the distributed permutation matrix:
    auto P = part(*A);
    auto R = transpose(*P);

    // Reorder the matrix:
    A = product(*R, *product(*A, *P));

    // and the RHS vector:
    std::vector<double> new_rhs(R->loc_rows());
    R->move_to_backend(typename DBackend::params());
    Alina::backend::spmv(1, *R, rhs, 0, new_rhs);
    rhs.swap(new_rhs);

    // Update the number of the local rows
    // (it may have changed as a result of permutation):
    chunk = A->loc_rows();
    prof.toc("partition");
  }
#endif

  // Initialize the solver:
  prof.tic("setup");
  Solver solve(world, A);
  prof.toc("setup");

  // Show the mini-report on the constructed solver:
  if (world.rank == 0)
    std::cout << solve << std::endl;

  // Solve the system with the zero initial approximation:
  std::vector<double> x(chunk, 0.0);

  prof.tic("solve");
  Alina::SolverResult r = solve(*A, rhs, x);
  prof.toc("solve");

  // Output the number of iterations, the relative error,
  // and the profiling data:
  if (world.rank == 0)
    std::cout << "Iters: " << r.nbIteration() << std::endl
              << "Error: " << r.residual() << std::endl
              << prof << std::endl;
}
