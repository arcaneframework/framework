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

#include "./SampleProblemCommon.h"

#include "TestMainMpi.h"

using namespace Arcane;

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
  // If we want to test dynamic backends:
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
  n = sample_problem_distributed(comm.rank, comm.size, matrix_size, 1, ptr, col, val, rhs);

  solve_scalar(comm, n, ptr, col, val, prm, rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
