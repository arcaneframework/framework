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
#include "arccore/alina/Coarsening.h"

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

int main(int argc, char* argv[])
{
  // The command line should contain the matrix, the RHS, and the coordinate files:
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <A.bin> <b.bin> <coo.bin>" << std::endl;
    return 1;
  }

  Alina::mpi_init mpi(&argc, &argv);
  Alina::mpi_communicator world(MPI_COMM_WORLD);

  // The profiler:
  auto& prof = Alina::Profiler::globalProfiler();

  // Read the system matrix, the RHS, and the coordinates:
  prof.tic("read");
  // Get the global size of the matrix:
  ptrdiff_t rows = Alina::IO::crs_size<ptrdiff_t>(argv[1]);

  // Split the matrix into approximately equal chunks of rows, and
  // make sure each chunk size is divisible by 3.
  ptrdiff_t chunk = (rows + world.size - 1) / world.size;
  if (chunk % 3)
    chunk += 3 - chunk % 3;

  ptrdiff_t row_beg = std::min(rows, chunk * world.rank);
  ptrdiff_t row_end = std::min(rows, row_beg + chunk);
  chunk = row_end - row_beg;

  // Read our part of the system matrix, the RHS and the coordinates.
  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val, rhs, coo;
  Alina::IO::read_crs(argv[1], rows, ptr, col, val, row_beg, row_end);

  ptrdiff_t n, m;
  Alina::IO::read_dense(argv[2], n, m, rhs, row_beg, row_end);
  Alina::precondition(n == rows && m == 1, "The RHS file has wrong dimensions");

  Alina::IO::read_dense(argv[3], n, m, coo, row_beg / 3, row_end / 3);
  Alina::precondition(n * 3 == rows && m == 3, "The coordinate file has wrong dimensions");
  prof.toc("read");

  if (world.rank == 0) {
    std::cout
    << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl
    << "RHS " << argv[2] << ": " << rows << "x1" << std::endl
    << "Coords " << argv[3] << ": " << rows / 3 << "x3" << std::endl;
  }

  // Declare the backends and the solver type
  typedef Alina::BuiltinBackend<double> SBackend; // the solver backend
  typedef Alina::BuiltinBackend<float> PBackend; // the preconditioner backend

  using Solver = DistributedPreconditionedSolver<DistributedAMG<
                                                 PBackend,
                                                 DistributedSmoothedAggregationCoarsening<PBackend>,
                                                 DistributedSPAI0Relaxation<PBackend>>,
                                                 DistributedConjugateGradientSolver<PBackend>>;

  // The distributed matrix
  auto A = std::make_shared<Alina::DistributedMatrix<SBackend>>(
  world, std::tie(chunk, ptr, col, val));

  // Partition the matrix, the RHS vector, and the coordinates.
  // If neither ParMETIS not PT-SCOTCH are not available,
  // just keep the current naive partitioning.
#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
  typedef Alina::ParmetisMatrixPartitioner<SBackend> Partition;

  if (world.size > 1) {
    auto t = prof.scoped_tic("partition");
    Partition part;

    // part(A) returns the distributed permutation matrix.
    // Keep the DOFs belonging to the same grid nodes together
    // (use block-wise partitioning with block size 3).
    auto P = part(*A, 3);
    auto R = transpose(*P);

    // Reorder the matrix:
    A = product(*R, *product(*A, *P));

    // Reorder the RHS vector and the coordinates:
    R->move_to_backend();
    std::vector<double> new_rhs(R->loc_rows());
    std::vector<double> new_coo(R->loc_rows());
    Alina::backend::spmv(1, *R, rhs, 0, new_rhs);
    Alina::backend::spmv(1, *R, coo, 0, new_coo);
    rhs.swap(new_rhs);
    coo.swap(new_coo);

    // Update the number of the local rows
    // (it may have changed as a result of permutation).
    chunk = A->loc_rows();
  }
#endif

  // Solver parameters:
  Solver::params prm;
  prm.solver.maxiter = 500;
  prm.precond.coarsening.aggr.eps_strong = 0;

  // Convert the coordinates to the rigid body modes.
  // The function returns the number of near null-space vectors
  // (3 in 2D case, 6 in 3D case) and writes the vectors to the
  // std::vector<double> specified as the last argument:
  prm.precond.coarsening.aggr.nullspace.cols = Alina::rigid_body_modes(3, coo, prm.precond.coarsening.aggr.nullspace.B);

  // Initialize the solver with the system matrix.
  prof.tic("setup");
  Solver solve(world, A, prm);
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
  if (world.rank == 0) {
    std::cout << "Iters: " << r.nbIteration() << std::endl
              << "Error: " << r.residual() << std::endl
              << prof << std::endl;
  }
}
