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
#include <string>

#include <boost/program_options.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#if defined(SOLVER_BACKEND_CUDA)
#  include "arccore/alina/CudaBackend.h"
#  include "arccore/alina/relaxation_cusparse_ilu0.h"
   typedef Arcane::Alina::backend::cuda<double> Backend;
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#include "arccore/alina/BuiltinBackend.h"
typedef Arcane::Alina::BuiltinBackend<double> Backend;
#endif

#if defined(SOLVER_BACKEND_BUILTIN)
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
#endif

#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/Relaxation.h"
#include "arccore/alina/CPRDynamicRowSumPreconditioner.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;

using Alina::precondition;

//---------------------------------------------------------------------------
template <class Matrix>
void solve_cpr(const Matrix& K, const std::vector<double>& rhs, Alina::PropertyTree& prm)
{
  auto& prof = Alina::Profiler::globalProfiler();
  Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
  vex::Context ctx(vex::Filter::Env);
  std::cout << ctx << std::endl;
  bprm.q = ctx;
#elif defined(SOLVER_BACKEND_VIENNACL)
  std::cout
  << viennacl::ocl::current_device().name()
  << " (" << viennacl::ocl::current_device().vendor() << ")\n\n";
#elif defined(SOLVER_BACKEND_CUDA)
  cusparseCreate(&bprm.cusparse_handle);
  {
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << prop.name << std::endl
              << std::endl;
  }
#endif

  auto t1 = prof.scoped_tic("CPR");

  typedef Alina::AMG<Backend, Alina::CoarseningRuntime, Alina::RelaxationRuntime>
  PPrecond;

  using SPrecond = Alina::RelaxationAsPreconditioner<Backend, Alina::RelaxationRuntime>;

  prof.tic("setup");
  Alina::PreconditionedSolver<
    Alina::preconditioner::CPRDynamicRowSumPreconditioner<PPrecond, SPrecond>,
    Alina::SolverRuntime<Backend>>
  solve(K, prm, bprm);
  prof.toc("setup");

  std::cout << solve.precond() << std::endl;

  auto f = Backend::copy_vector(rhs, bprm);
  auto x = Backend::create_vector(rhs.size(), bprm);
  Alina::backend::clear(*x);

  prof.tic("solve");
  Alina::SolverResult r = solve(*f, *x);
  prof.toc("solve");

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl;
}

#if defined(SOLVER_BACKEND_BUILTIN)
//---------------------------------------------------------------------------
template <int B, class Matrix>
void solve_block_cpr(const Matrix& K, const std::vector<double>& rhs, Alina::PropertyTree& prm)
{
  auto& prof = Alina::Profiler::globalProfiler();
  auto t1 = prof.scoped_tic("CPR");

  typedef Alina::StaticMatrix<double, B, B> val_type;
  typedef Alina::StaticMatrix<double, B, 1> rhs_type;

  typedef Alina::BuiltinBackend<val_type> SBackend;
  typedef Alina::BuiltinBackend<double> PBackend;

  using PPrecond = Alina::AMG<PBackend, Alina::CoarseningRuntime, Alina::RelaxationRuntime>;
  using SPrecond = Alina::RelaxationAsPreconditioner<SBackend, Alina::RelaxationRuntime>;

  typename SBackend::params bprm;

  prof.tic("setup");
  Alina::PreconditionedSolver<
  Alina::preconditioner::CPRDynamicRowSumPreconditioner<PPrecond, SPrecond>,
  Alina::SolverRuntime<SBackend>>
  solve(Alina::adapter::block_matrix<val_type>(K), prm, bprm);
  prof.toc("setup");

  std::cout << solve.precond() << std::endl;

  size_t n = Alina::backend::nbRow(K) / B;
  auto rhs_ptr = reinterpret_cast<const rhs_type*>(rhs.data());

  SmallSpan<const rhs_type> f(rhs_ptr, n);

  auto x = SBackend::create_vector(n, bprm);
  Alina::backend::clear(*x);

  prof.tic("solve");
  Alina::SolverResult r = solve(f, *x);
  prof.toc("solve");

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl;
}
#endif

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();
  using Alina::precondition;
  using std::string;
  using std::vector;

  namespace po = boost::program_options;
  namespace io = Alina::IO;

  po::options_description desc("Options");

  desc.add_options()("help,h", "show help")(
  "binary,B",
  po::bool_switch()->default_value(false),
  "When specified, treat input files as binary instead of as MatrixMarket. "
  "It is assumed the files were converted to binary format with mm2bin utility. ")(
  "matrix,A",
  po::value<string>()->required(),
  "The system matrix in MatrixMarket format")(
  "rhs,f",
  po::value<string>(),
  "The right-hand side in MatrixMarket format")(
  "weights,w",
  po::value<string>(),
  "Equation weights in MatrixMarket format")(
  "runtime-block-size,b",
  po::value<int>(),
  "The block size of the system matrix set at runtime")(
  "static-block-size,c",
  po::value<int>()->default_value(1),
  "The block size of the system matrix set at compiletime")(
  "params,P",
  po::value<string>(),
  "parameter file in json format")(
  "prm,p",
  po::value<vector<string>>()->multitoken(),
  "Parameters specified as name=value pairs. "
  "May be provided multiple times. Examples:\n"
  "  -p solver.tol=1e-3\n"
  "  -p precond.coarse_enough=300");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  po::notify(vm);

  Alina::PropertyTree prm;
  if (vm.count("params"))
    prm.read_json(vm["params"].as<string>());

  if (vm.count("prm")) {
    for (const string& v : vm["prm"].as<vector<string>>()) {
      prm.putKeyValue(v);
    }
  }

  int cb = vm["static-block-size"].as<int>();

  if (vm.count("runtime-block-size"))
    prm.put("precond.block_size", vm["runtime-block-size"].as<int>());
  else
    prm.put("precond.block_size", cb);

  size_t rows;
  vector<ptrdiff_t> ptr, col;
  vector<double> val, rhs, wgt;
  std::vector<char> pm;

  {
    auto t = prof.scoped_tic("reading");

    string Afile = vm["matrix"].as<string>();
    bool binary = vm["binary"].as<bool>();

    if (binary) {
      io::read_crs(Afile, rows, ptr, col, val);
    }
    else {
      size_t cols;
      std::tie(rows, cols) = io::mm_reader(Afile)(ptr, col, val);
      precondition(rows == cols, "Non-square system matrix");
    }

    if (vm.count("rhs")) {
      string bfile = vm["rhs"].as<string>();

      size_t n, m;

      if (binary) {
        io::read_dense(bfile, n, m, rhs);
      }
      else {
        std::tie(n, m) = io::mm_reader(bfile)(rhs);
      }

      precondition(n == rows && m == 1, "The RHS vector has wrong size");
    }
    else {
      rhs.resize(rows, 1.0);
    }

    if (vm.count("weights")) {
      string wfile = vm["weights"].as<string>();

      size_t n, m;

      if (binary) {
        io::read_dense(wfile, n, m, wgt);
      }
      else {
        std::tie(n, m) = io::mm_reader(wfile)(wgt);
      }

      prm.put("precond.weights", &wgt[0]);
      prm.put("precond.weights_size", wgt.size());
    }
  }

#define CALL_BLOCK_SOLVER(z, data, B) \
  case B: \
    solve_block_cpr<B>(std::tie(rows, ptr, col, val), rhs, prm); \
    break;

  switch (cb) {
  case 1:
    solve_cpr(std::tie(rows, ptr, col, val), rhs, prm);
    break;

#if defined(SOLVER_BACKEND_BUILTIN) || defined(SOLVER_BACKEND_VEXCL)
    BOOST_PP_SEQ_FOR_EACH(CALL_BLOCK_SOLVER, ~, ARCCORE_ALINA_BLOCK_SIZES)
#endif

  default:
    precondition(false, "Unsupported block size");
    break;
  }

  std::cout << prof << std::endl;
}
