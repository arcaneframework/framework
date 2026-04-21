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

#include <iostream>
#include <vector>
#include <string>

#include <boost/program_options.hpp>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedPreconditionedSolver.h"
#include "arccore/alina/DistributedPreconditioner.h"
#include "arccore/alina/DistributedSolverRuntime.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

#include "arcane/utils/Exception.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/launcher/ArcaneLauncher.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IProfilingService.h"

#include "AlinaSamplesCommon.h"

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
template <class Backend, class Matrix>
std::shared_ptr<Alina::DistributedMatrix<Backend>>
partition(Alina::mpi_communicator comm, const Matrix& Astrip,
          typename Backend::vector& rhs, const typename Backend::params& bprm,
          Alina::eMatrixPartitionerType ptype, int block_size = 1)
{
  auto& prof = Alina::Profiler::globalProfiler();
  typedef typename Backend::value_type val_type;
  typedef typename Alina::math::rhs_of<val_type>::type rhs_type;
  typedef Alina::DistributedMatrix<Backend> DMatrix;

  auto A = std::make_shared<DMatrix>(comm, Astrip);

  if (comm.size == 1 || ptype == Alina::eMatrixPartitionerType::merge)
    return A;

  prof.tic("partition");
  Alina::PropertyTree prm;
  prm.put("type", ptype);
  Alina::MatrixPartitionerRuntime<Backend> part(prm);

  auto I = part(*A, block_size);
  auto J = transpose(*I);
  A = product(*J, *product(*A, *I));

  Alina::numa_vector<rhs_type> new_rhs(J->loc_rows());

  J->move_to_backend(bprm);

  Alina::backend::spmv(1, *J, rhs, 0, new_rhs);
  rhs.swap(new_rhs);
  prof.toc("partition");

  return A;
}

//---------------------------------------------------------------------------
void solve_scalar(Alina::mpi_communicator comm,
                  ptrdiff_t chunk,
                  const std::vector<ptrdiff_t>& ptr,
                  const std::vector<ptrdiff_t>& col,
                  const std::vector<double>& val,
                  const Alina::PropertyTree& prm,
                  const std::vector<double>& f,
                  Alina::eMatrixPartitionerType ptype)
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

  auto get_distributed_matrix = [&]() {
    auto t = prof.scoped_tic("distributed matrix");
    std::shared_ptr<DMatrix> A;

    if (ptype != Alina::eMatrixPartitionerType::merge) {
      A = partition<Backend>(comm,
                             std::tie(chunk, ptr, col, val), rhs, bprm, ptype,
                             prm.get("precond.coarsening.aggr.block_size", 1));
      chunk = A->loc_rows();
    }
    else {
      A = std::make_shared<DMatrix>(comm, std::tie(chunk, ptr, col, val));
    }

    return A;
  };

  std::shared_ptr<DMatrix> A;
  std::shared_ptr<Solver> solve;

  {
    auto t = prof.scoped_tic("setup");
    A = get_distributed_matrix();
    solve = std::make_shared<Solver>(comm, A, prm, bprm);
    Alina::PropertyTree prm2;
    solve->prm.get(prm2);
    std::cout << "SOLVER parameters=" << prm2 << "\n";
  }

  if (comm.rank == 0) {
    std::cout << "SolverInfo:\n";
    std::cout << *solve << std::endl;
  }

  if (prm.get("precond.allow_rebuild", false)) {
    if (comm.rank == 0) {
      std::cout << "Rebuilding the preconditioner..." << std::endl;
    }

    {
      auto t = prof.scoped_tic("rebuild");
      A = get_distributed_matrix();
      solve->precond().rebuild(A);
    }

    if (comm.rank == 0) {
      std::cout << *solve << std::endl;
    }
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

//---------------------------------------------------------------------------
int main2(const Alina::SampleMainContext& ctx, int argc, char* argv[])
{
  ITraceMng* tm = ctx.traceMng();
  auto& prof = Alina::Profiler::globalProfiler();

  //Alina::mpi_init_thread mpi(&argc, &argv);
  Alina::mpi_communicator comm(MPI_COMM_WORLD);

  tm->info() << "World size: " << comm.size;

  // Read configuration from command line
  namespace po = boost::program_options;
  po::options_description desc("Options");

  auto default_partitioner_type = Alina::eMatrixPartitionerType::merge;
#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
  default_partitioner_type = Alina::eMatrixPartitionerType::parmetis;
#endif

  desc.add_options()("help,h", "show help")("matrix,A",
                                            po::value<std::string>(),
                                            "System matrix in the MatrixMarket format. "
                                            "When not specified, a Poisson problem in 3D unit cube is assembled. ");
  desc.add_options()("partitioner,r",
                     po::value<Alina::eMatrixPartitionerType>()->default_value(
                     default_partitioner_type),
                     "Repartition the system matrix");
  desc.add_options()("size,n",
                     po::value<ptrdiff_t>()->default_value(32),
                     "domain size");
  desc.add_options()("prm-file,P", po::value<std::string>(),
                     "Parameter file in json format. ");
  desc.add_options()("prm,p",
                     po::value<std::vector<std::string>>()->multitoken(),
                     "Parameters specified as name=value pairs. "
                     "May be provided multiple times. Examples:\n"
                     "  -p solver.tol=1e-3\n"
                     "  -p precond.coarse_enough=300");
  desc.add_options()("test-rebuild",
                     po::bool_switch()->default_value(false),
                     "When specified, try to rebuild the solver before solving. ");

  po::positional_options_description p;
  p.add("prm", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    if (comm.rank == 0)
      std::cout << desc << std::endl;
    return 0;
  }

  Alina::PropertyTree prm;
  if (vm.count("prm-file")) {
    prm.read_json(vm["prm-file"].as<std::string>());
  }

  if (vm.count("prm")) {
    for (const std::string& v : vm["prm"].as<std::vector<std::string>>()) {
      tm->info() << "PUT_KEY_VALUE v=" << v;
      prm.putKeyValue(v);
    }
  }

  ptrdiff_t n;
  std::vector<ptrdiff_t> ptr;
  std::vector<ptrdiff_t> col;
  std::vector<double> val;
  std::vector<double> rhs;

  Alina::eMatrixPartitionerType ptype = vm["partitioner"].as<Alina::eMatrixPartitionerType>();

  prof.tic("assemble");
  Int64 matrix_size = vm["size"].as<ptrdiff_t>();
  tm->info() << "Matrix size=" << matrix_size;
  n = assemble_poisson3d(comm, matrix_size, 1, ptr, col, val, rhs);
  prof.toc("assemble");

  if (vm["test-rebuild"].as<bool>()) {
    prm.put("precond.allow_rebuild", true);
  }

  solve_scalar(comm, n, ptr, col, val, prm, rhs, ptype);
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  return Arcane::Alina::SampleMainContext::execMain(main2, argc, argv);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
