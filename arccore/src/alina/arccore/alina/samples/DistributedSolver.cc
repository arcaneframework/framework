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

// Pour Eigen
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <iostream>
#include <vector>
#include <string>

#include <boost/program_options.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"

#if defined(SOLVER_BACKEND_CUDA)
#  include "arccore/alina/CudaBackend.h"
#  include "arccore/alina/relaxation_cusparse_ilu0.h"
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#endif

#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/DistributedPreconditionedSolver.h"
#include "arccore/alina/DistributedPreconditioner.h"
#include "arccore/alina/DistributedSolverRuntime.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

#ifndef ARCCORE_ALINA_BLOCK_SIZES
#  define ARCCORE_ALINA_BLOCK_SIZES (3)(4)
#endif

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
ptrdiff_t
read_matrix_market(Alina::mpi_communicator comm,
                   const std::string& A_file, const std::string& rhs_file, int block_size,
                   std::vector<ptrdiff_t>& ptr,
                   std::vector<ptrdiff_t>& col,
                   std::vector<double>& val,
                   std::vector<double>& rhs)
{
  Alina::IO::mm_reader A_mm(A_file);
  ptrdiff_t n = A_mm.rows();

  ptrdiff_t chunk = (n + comm.size - 1) / comm.size;
  if (chunk % block_size != 0) {
    chunk += block_size - chunk % block_size;
  }

  ptrdiff_t row_beg = std::min(n, chunk * comm.rank);
  ptrdiff_t row_end = std::min(n, row_beg + chunk);

  chunk = row_end - row_beg;

  A_mm(ptr, col, val, row_beg, row_end);

  if (rhs_file.empty()) {
    rhs.resize(chunk);
    std::fill(rhs.begin(), rhs.end(), 1.0);
  }
  else {
    Alina::IO::mm_reader rhs_mm(rhs_file);
    rhs_mm(rhs, row_beg, row_end);
  }

  return chunk;
}

//---------------------------------------------------------------------------
ptrdiff_t
read_binary(Alina::mpi_communicator comm,
            const std::string& A_file, const std::string& rhs_file, int block_size,
            std::vector<ptrdiff_t>& ptr,
            std::vector<ptrdiff_t>& col,
            std::vector<double>& val,
            std::vector<double>& rhs)
{
  ptrdiff_t n = Alina::IO::crs_size<ptrdiff_t>(A_file);

  ptrdiff_t chunk = (n + comm.size - 1) / comm.size;
  if (chunk % block_size != 0) {
    chunk += block_size - chunk % block_size;
  }

  ptrdiff_t row_beg = std::min(n, chunk * comm.rank);
  ptrdiff_t row_end = std::min(n, row_beg + chunk);

  chunk = row_end - row_beg;

  Alina::IO::read_crs(A_file, n, ptr, col, val, row_beg, row_end);

  if (rhs_file.empty()) {
    rhs.resize(chunk);
    std::fill(rhs.begin(), rhs.end(), 1.0);
  }
  else {
    ptrdiff_t rows, cols;
    Alina::IO::read_dense(rhs_file, rows, cols, rhs, row_beg, row_end);
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

#if defined(SOLVER_BACKEND_BUILTIN)
  Alina::numa_vector<rhs_type> new_rhs(J->loc_rows());
#elif defined(SOLVER_BACKEND_CUDA)
  thrust::device_vector<rhs_type> new_rhs(J->loc_rows());
#endif

  J->move_to_backend(bprm);

  Alina::backend::spmv(1, *J, rhs, 0, new_rhs);
  rhs.swap(new_rhs);
  prof.toc("partition");

  return A;
}

//---------------------------------------------------------------------------
#if defined(SOLVER_BACKEND_BUILTIN)
template <int B>
void solve_block(Alina::mpi_communicator comm,
                 ptrdiff_t chunk,
                 const std::vector<ptrdiff_t>& ptr,
                 const std::vector<ptrdiff_t>& col,
                 const std::vector<double>& val,
                 const Alina::PropertyTree& prm,
                 const std::vector<double>& f,
                 Alina::eMatrixPartitionerType ptype)
{
  auto& prof = Alina::Profiler::globalProfiler();
  typedef Alina::StaticMatrix<double, B, B> val_type;
  typedef Alina::StaticMatrix<double, B, 1> rhs_type;

  typedef Alina::BuiltinBackend<val_type> Backend;

  typedef Alina::DistributedMatrix<Backend> DMatrix;

  typedef Alina::DistributedPreconditionedSolver<
  Alina::DistributedPreconditioner<Backend>,
  Alina::DistributedSolverRuntime<Backend>>
  Solver;

  typename Backend::params bprm;

  Alina::numa_vector<rhs_type> rhs(reinterpret_cast<const rhs_type*>(&f[0]),
                                            reinterpret_cast<const rhs_type*>(&f[0]) + chunk / B);

  auto get_distributed_matrix = [&]() {
    auto t = prof.scoped_tic("distributed matrix");

    std::shared_ptr<DMatrix> A;

    if (ptype != Alina::eMatrixPartitionerType::merge) {
      A = partition<Backend>(comm,
                             Alina::adapter::block_matrix<val_type>(std::tie(chunk, ptr, col, val)),
                             rhs, bprm, ptype, prm.get("precond.coarsening.aggr.block_size", 1));
      chunk = A->loc_rows();
    }
    else {
      A = std::make_shared<DMatrix>(
      comm,
      Alina::adapter::block_matrix<val_type>(std::tie(chunk, ptr, col, val)));
    }

    return A;
  };

  std::shared_ptr<DMatrix> A;
  std::shared_ptr<Solver> solve;

  {
    auto t = prof.scoped_tic("setup");
    A = get_distributed_matrix();
    solve = std::make_shared<Solver>(comm, A, prm, bprm);
  }

  if (comm.rank == 0) {
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

  Alina::numa_vector<rhs_type> x(chunk);

  prof.tic("solve");
  Alina::SolverResult r = (*solve)(rhs, x);
  prof.toc("solve");

  if (comm.rank == 0) {
    std::cout << "Iterations: " << r.nbIteration() << std::endl
              << "Error:      " << r.residual() << std::endl
              << prof << std::endl;
  }
}
#endif

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
#if defined(SOLVER_BACKEND_BUILTIN)
  //using Backend = Alina::BuiltinBackend<double>;
  using Backend = Alina::BuiltinBackend<double, Arcane::Int32>;
#elif defined(SOLVER_BACKEND_CUDA)
  using Backend = Alina::backend::cuda<double>;
#endif

  std::cout << "Using scalar solve ptr_size=" << sizeof(ptrdiff_t)
            << " ptr_type_size=" << sizeof(Backend::ptr_type)
            << " col_type_size=" << sizeof(Backend::col_type)
            << " value_type_size=" << sizeof(Backend::value_type)
            << "\n";

  typedef Alina::DistributedMatrix<Backend> DMatrix;

  using Solver = Alina::DistributedPreconditionedSolver<Alina::DistributedPreconditioner<Backend>, Alina::DistributedSolverRuntime<Backend>>;

  typename Backend::params bprm;

#if defined(SOLVER_BACKEND_BUILTIN)
  Alina::numa_vector<double> rhs(f);
#elif defined(SOLVER_BACKEND_CUDA)
  cusparseCreate(&bprm.cusparse_handle);
  thrust::device_vector<double> rhs(f);
#endif

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

#if defined(SOLVER_BACKEND_BUILTIN)
  Alina::numa_vector<double> x(chunk);
#elif defined(SOLVER_BACKEND_CUDA)
  thrust::device_vector<double> x(chunk, 0.0);
#endif

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
int main(int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();

  Alina::mpi_init_thread mpi(&argc, &argv);
  Alina::mpi_communicator comm(MPI_COMM_WORLD);

  if (comm.rank == 0)
    std::cout << "World size: " << comm.size << std::endl;

  // Read configuration from command line
  namespace po = boost::program_options;
  po::options_description desc("Options");

  desc.add_options()("help,h", "show help")("matrix,A",
                                            po::value<std::string>(),
                                            "System matrix in the MatrixMarket format. "
                                            "When not specified, a Poisson problem in 3D unit cube is assembled. ")(
  "rhs,f",
  po::value<std::string>()->default_value(""),
  "The RHS vector in the MatrixMarket format. "
  "When omitted, a vector of ones is used by default. "
  "Should only be provided together with a system matrix. ")(
  "Ap",
  po::value<std::vector<std::string>>()->multitoken(),
  "Pre-partitioned matrix (single file per MPI process)")(
  "fp",
  po::value<std::vector<std::string>>()->multitoken(),
  "Pre-partitioned RHS (single file per MPI process)")(
  "binary,B",
  po::bool_switch()->default_value(false),
  "When specified, treat input files as binary instead of as MatrixMarket. "
  "It is assumed the files were converted to binary format with mm2bin utility. ")(
  "block-size,b",
  po::value<int>()->default_value(1),
  "The block size of the system matrix. "
  "When specified, the system matrix is assumed to have block-wise structure. "
  "This usually is the case for problems in elasticity, structural mechanics, "
  "for coupled systems of PDE (such as Navier-Stokes equations), etc. ")(
  "partitioner,r",
  po::value<Alina::eMatrixPartitionerType>()->default_value(
#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
  Alina::eMatrixPartitionerType::parmetis
#else
  Alina::eMatrixPartitionerType::merge
#endif
  ),
  "Repartition the system matrix")(
  "size,n",
  po::value<ptrdiff_t>()->default_value(128),
  "domain size")("prm-file,P",
                 po::value<std::string>(),
                 "Parameter file in json format. ")(
  "prm,p",
  po::value<std::vector<std::string>>()->multitoken(),
  "Parameters specified as name=value pairs. "
  "May be provided multiple times. Examples:\n"
  "  -p solver.tol=1e-3\n"
  "  -p precond.coarse_enough=300")(
  "test-rebuild",
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
      prm.putKeyValue(v);
    }
  }

  ptrdiff_t n;
  std::vector<ptrdiff_t> ptr;
  std::vector<ptrdiff_t> col;
  std::vector<double> val;
  std::vector<double> rhs;

  int block_size = vm["block-size"].as<int>();
  int aggr_block = prm.get("precond.coarsening.aggr.block_size", 1);

  bool binary = vm["binary"].as<bool>();
  Alina::eMatrixPartitionerType ptype = vm["partitioner"].as<Alina::eMatrixPartitionerType>();

  if (vm.count("matrix")) {
    prof.tic("read");
    if (binary) {
      n = read_binary(comm,
                      vm["matrix"].as<std::string>(),
                      vm["rhs"].as<std::string>(),
                      block_size * aggr_block, ptr, col, val, rhs);
    }
    else {
      n = read_matrix_market(comm,
                             vm["matrix"].as<std::string>(),
                             vm["rhs"].as<std::string>(),
                             block_size * aggr_block, ptr, col, val, rhs);
    }
    prof.toc("read");
  }
  else if (vm.count("Ap")) {
    prof.tic("read");
    ptype = Alina::eMatrixPartitionerType::merge;

    std::vector<std::string> Aparts = vm["Ap"].as<std::vector<std::string>>();
    comm.check(Aparts.size() == static_cast<size_t>(comm.size),
               "--Ap should have single entry per MPI process");

    if (binary) {
      Alina::IO::read_crs(Aparts[comm.rank], n, ptr, col, val);
    }
    else {
      ptrdiff_t m;
      std::tie(n, m) = Alina::IO::mm_reader(Aparts[comm.rank])(ptr, col, val);
    }

    if (vm.count("fp")) {
      std::vector<std::string> fparts = vm["fp"].as<std::vector<std::string>>();
      comm.check(fparts.size() == static_cast<size_t>(comm.size),
                 "--fp should have single entry per MPI process");

      ptrdiff_t rows;
      ptrdiff_t cols;

      if (binary) {
        Alina::IO::read_dense(fparts[comm.rank], rows, cols, rhs);
      }
      else {
        std::tie(rows, cols) = Alina::IO::mm_reader(fparts[comm.rank])(rhs);
      }

      comm.check(rhs.size() == static_cast<size_t>(n), "Wrong RHS size");
    }
    else {
      rhs.resize(n, 1);
    }
    prof.toc("read");
  }
  else {
    prof.tic("assemble");
    n = assemble_poisson3d(comm,
                           vm["size"].as<ptrdiff_t>(),
                           block_size * aggr_block, ptr, col, val, rhs);
    prof.toc("assemble");
  }

  if (vm["test-rebuild"].as<bool>()) {
    prm.put("precond.allow_rebuild", true);
  }

  switch (block_size) {

#if defined(SOLVER_BACKEND_BUILTIN)
#define ARCCORE_ALINA_CALL_BLOCK_SOLVER(z, data, B) \
  case B: \
    solve_block<B>(comm, n, ptr, col, val, prm, rhs, ptype); \
    break;

    BOOST_PP_SEQ_FOR_EACH(ARCCORE_ALINA_CALL_BLOCK_SOLVER, ~, ARCCORE_ALINA_BLOCK_SIZES)

#undef ARCCORE_ALINA_CALL_BLOCK_SOLVER
#endif

  case 1:
    solve_scalar(comm, n, ptr, col, val, prm, rhs, ptype);
    break;
  default:
    if (comm.rank == 0)
      std::cout << "Unsupported block size!" << std::endl;
  }
}
