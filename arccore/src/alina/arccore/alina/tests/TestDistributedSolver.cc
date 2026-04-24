// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedPreconditionedSolver.h"
#include "arccore/alina/DistributedPreconditioner.h"
#include "arccore/alina/DistributedSolverRuntime.h"

#include "TestMainMpi.h"

using namespace Arcane;

namespace math = Alina::math;

//---------------------------------------------------------------------------
ptrdiff_t
assemble_poisson3d(Alina::mpi_communicator comm,
                   ptrdiff_t n, int block_size,
                   std::vector<ptrdiff_t>& ptr,
                   std::vector<ptrdiff_t>& col,
                   std::vector<double>& val,
                   std::vector<double>& rhs)
{
  ptrdiff_t n3 = n * n * n;

  ptrdiff_t chunk = (n3 + comm.size - 1) / comm.size;
  if (chunk % block_size != 0) {
    chunk += block_size - chunk % block_size;
  }
  ptrdiff_t row_beg = std::min(n3, chunk * comm.rank);
  ptrdiff_t row_end = std::min(n3, row_beg + chunk);
  chunk = row_end - row_beg;

  ptr.clear();
  ptr.reserve(chunk + 1);
  col.clear();
  col.reserve(chunk * 7);
  val.clear();
  val.reserve(chunk * 7);

  rhs.resize(chunk);
  std::fill(rhs.begin(), rhs.end(), 1.0);

  const double h2i = (n - 1) * (n - 1);
  ptr.push_back(0);

  for (ptrdiff_t idx = row_beg; idx < row_end; ++idx) {
    ptrdiff_t k = idx / (n * n);
    ptrdiff_t j = (idx / n) % n;
    ptrdiff_t i = idx % n;

    if (k > 0) {
      col.push_back(idx - n * n);
      val.push_back(-h2i);
    }

    if (j > 0) {
      col.push_back(idx - n);
      val.push_back(-h2i);
    }

    if (i > 0) {
      col.push_back(idx - 1);
      val.push_back(-h2i);
    }

    col.push_back(idx);
    val.push_back(6 * h2i);

    if (i + 1 < n) {
      col.push_back(idx + 1);
      val.push_back(-h2i);
    }

    if (j + 1 < n) {
      col.push_back(idx + n);
      val.push_back(-h2i);
    }

    if (k + 1 < n) {
      col.push_back(idx + n * n);
      val.push_back(-h2i);
    }

    ptr.push_back(col.size());
  }

  return chunk;
}

//---------------------------------------------------------------------------

void solve_scalar(Alina::mpi_communicator comm,
                  ptrdiff_t chunk,
                  const std::vector<ptrdiff_t>& ptr,
                  const std::vector<ptrdiff_t>& col,
                  const std::vector<double>& val,
                  const Alina::PropertyTree& prm,
                  const std::vector<double>& f)
{
  auto& prof = Alina::Profiler::globalProfiler();
  //using Backend = Alina::BuiltinBackend<double>;

  using BackendValueType = double;
  using Backend = Alina::BuiltinBackend<BackendValueType, Arcane::Int32>;

  std::cout << "Using scalar solve ptr_size=" << sizeof(ptrdiff_t)
            << " ptr_type_size=" << sizeof(Backend::ptr_type)
            << " col_type_size=" << sizeof(Backend::col_type)
            << " value_type_size=" << sizeof(Backend::value_type)
            << "\n";

  typedef Alina::DistributedMatrix<Backend> DMatrix;

  using CoarseningType = Alina::DistributedSmoothedAggregationCoarsening<Backend>;
  using RelaxationType = Alina::DistributedSPAI0Relaxation<Backend>;
  // Si on veut tester les backend dynamiques:
  //using CoarseningType = Alina::DistributedCoarseningRuntime<Backend>;
  //using RelaxationType = Alina::DistributedRelaxationRuntime<Backend>,

  using AMGPrecondType = Alina::DistributedAMG<Backend, CoarseningType, RelaxationType,
                                               Alina::DistributedDirectSolverRuntime<Backend>,
                                               Alina::MatrixPartitionerRuntime<Backend>>;

  using Solver = Alina::DistributedPreconditionedSolver<AMGPrecondType, Alina::DistributedSolverRuntime<Backend>>;

  typename Backend::params bprm;

  Alina::numa_vector<double> rhs(f);

  std::shared_ptr<DMatrix> A;
  std::shared_ptr<Solver> solve;

  {
    auto t = prof.scoped_tic("setup");
    A = std::make_shared<DMatrix>(comm, std::tie(chunk, ptr, col, val));
    solve = std::make_shared<Solver>(comm, A, prm, bprm);
    Alina::PropertyTree prm2;
    solve->prm.get(prm2);
    std::cout << "SOLVER parameters=" << prm2 << "\n";
  }

  if (comm.rank == 0) {
    std::cout << "SolverInfo:\n";
    std::cout << *solve << std::endl;
  }

  Alina::numa_vector<double> x(chunk);

  prof.tic("solve");
  Alina::SolverResult r = (*solve)(rhs, x);
  prof.toc("solve");

  if (comm.rank == 0) {
    std::cout << "Iterations: " << r.nbIteration() << std::endl
              << "Error:      " << r.residual() << std::endl
              << prof << std::endl;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(alina_test_mpi, BasicSolver)
{
  Alina::mpi_communicator comm(AlinaTest::global_mpi_comm_world);

  std::cout << "World size: " << comm.size << "\n";

  Alina::PropertyTree prm;

  ptrdiff_t n;
  std::vector<ptrdiff_t> ptr;
  std::vector<ptrdiff_t> col;
  std::vector<double> val;
  std::vector<double> rhs;

  Int64 matrix_size = 32;
  std::cout << "Matrix size=" << matrix_size << "\n";
  n = assemble_poisson3d(comm, matrix_size, 1, ptr, col, val, rhs);

  solve_scalar(comm, n, ptr, col, val, prm, rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
