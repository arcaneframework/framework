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
#include "arccore/alina/StaticMatrix.h"
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

// Block size
const int B = 3;

using namespace Arcane;
using namespace Arcane::Alina;

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // The command line should contain the matrix file name:
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix.bin>" << std::endl;
    return 1;
  }

  Alina::mpi_init mpi(&argc, &argv);
  Alina::mpi_communicator world(MPI_COMM_WORLD);

  auto& prof = Alina::Profiler::globalProfiler();

  prof.tic("read");
  // Get the global size of the matrix:
  ptrdiff_t rows = Alina::IO::crs_size<ptrdiff_t>(argv[1]);

  // Split the matrix into approximately equal chunks of rows, and
  // make sure each chunk size is divisible by the block size.
  ptrdiff_t chunk = (rows + world.size - 1) / world.size;
  if (chunk % B)
    chunk += B - chunk % B;

  ptrdiff_t row_beg = std::min(rows, chunk * world.rank);
  ptrdiff_t row_end = std::min(rows, row_beg + chunk);
  chunk = row_end - row_beg;

  // Read our part of the system matrix.
  std::vector<ptrdiff_t> ptr, col;
  std::vector<double> val;
  Alina::IO::read_crs(argv[1], rows, ptr, col, val, row_beg, row_end);
  prof.toc("read");

  if (world.rank == 0)
    std::cout
    << "World size: " << world.size << std::endl
    << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl;

  // Declare the backend and the solver types
  typedef Alina::StaticMatrix<double, B, B> dmat_type;
  typedef Alina::StaticMatrix<double, B, 1> dvec_type;
  typedef Alina::StaticMatrix<float, B, B> fmat_type;
  typedef Alina::BuiltinBackend<dmat_type> DBackend;
  typedef Alina::BuiltinBackend<fmat_type> FBackend;

  typedef DistributedPreconditionedSolver<
    DistributedAMG<
      FBackend,
      Alina::DistributedSmoothedAggregationCoarsening<FBackend>,
      Alina::DistributedSPAI0Relaxation<FBackend>>,
    Alina::DistributedBiCGStabSolver<DBackend>>
  Solver;

  // Solver parameters
  Solver::params prm;
  prm.solver.maxiter = 200;

  // We need to scale the matrix, so that it has the unit diagonal.
  // Since we only have the local rows for the matrix, and we may need the
  // remote diagonal values, it is more convenient to represent the scaling
  // with the matrix-matrix product (As = D^-1/2 A D^-1/2).
  prof.tic("scale");
  // Find the local diagonal values,
  // and form the CRS arrays for a diagonal matrix.
  std::vector<double> dia(chunk, 1.0);
  std::vector<ptrdiff_t> d_ptr(chunk + 1), d_col(chunk);
  for (ptrdiff_t i = 0, I = row_beg; i < chunk; ++i, ++I) {
    d_ptr[i] = i;
    d_col[i] = I;
    for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
      if (col[j] == I) {
        dia[i] = 1 / sqrt(val[j]);
        break;
      }
    }
  }
  d_ptr.back() = chunk;

  // Create the distributed diagonal matrix:
  Alina::DistributedMatrix<DBackend> D(world,
                                            Alina::adapter::block_matrix<dmat_type>(
                                            std::tie(chunk, d_ptr, d_col, dia)));

  // The scaled matrix is formed as product D * A * D,
  // where A is the local chunk of the matrix
  // converted to the block format on the fly.
  auto A = product(D, *product(Alina::DistributedMatrix<DBackend>(world, Alina::adapter::block_matrix<dmat_type>(std::tie(chunk, ptr, col, val))), D));
  prof.toc("scale");

  // Since the RHS in this case is filled with ones,
  // the scaled RHS is equal to dia.
  // Reinterpret the pointer to dia data to get the RHS in the block format:
  auto f_ptr = reinterpret_cast<dvec_type*>(dia.data());
  std::vector<dvec_type> rhs(f_ptr, f_ptr + chunk / B);

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
    std::vector<dvec_type> new_rhs(R->loc_rows());
    R->move_to_backend();
    Alina::backend::spmv(1, *R, rhs, 0, new_rhs);
    rhs.swap(new_rhs);

    // Update the number of the local rows
    // (it may have changed as a result of permutation).
    // Note that A->loc_rows() returns the number of blocks,
    // as the matrix uses block values.
    chunk = A->loc_rows();
    prof.toc("partition");
  }
#endif

  // Initialize the solver:
  prof.tic("setup");
  Solver solve(world, A, prm);
  prof.toc("setup");

  // Show the mini-report on the constructed solver:
  if (world.rank == 0)
    std::cout << solve << std::endl;

  // Solve the system with the zero initial approximation:
  std::vector<dvec_type> x(chunk, Alina::math::zero<dvec_type>());

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
