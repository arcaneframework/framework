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
#include <random>

// To remove warnings about deprecated Eigen usage.
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <boost/program_options.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#if defined(SOLVER_BACKEND_CUDA)
#include "arccore/alina/CudaBackend.h"
#include "arccore/alina/relaxation_cusparse_ilu0.h"
typedef Arcane::Alina::backend::cuda<double> Backend;
#elif defined(SOLVER_BACKEND_EIGEN)
#include "arccore/alina/EigenBackend.h"
typedef Arcane::Alina::backend::EigenBackend<double> Backend;
#else
#ifndef SOLVER_BACKEND_BUILTIN
#define SOLVER_BACKEND_BUILTIN
#endif
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
// Use 32 bit indexing for backend.
using Backend = Arcane::Alina::BuiltinBackend<double, Arcane::Int32>;
//using Backend = Arcane::Alina::BuiltinBackend<double>;
#endif

#include <arcane/utils/PlatformUtils.h>
#include <arcane/utils/String.h>
#include <arcane/utils/Convert.h>

#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/PreconditionerRuntime.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/IO.h"

#include "arccore/alina/Profiler.h"

#include "SampleProblemCommon.h"

#ifndef ARCCORE_ALINA_BLOCK_SIZES
#define ARCCORE_ALINA_BLOCK_SIZES (3)(4)
#endif

using namespace Arcane;

using Alina::precondition;

#ifdef SOLVER_BACKEND_BUILTIN
extern "C++" void
_doHypreSolver(int nb_row,
               std::vector<ptrdiff_t> const& ptr,
               std::vector<ptrdiff_t> const& col,
               std::vector<double> const& val,
               std::vector<double> const& rhs,
               std::vector<double>& x,
               int argc, char* argv[]);
#endif

#ifdef SOLVER_BACKEND_BUILTIN
//---------------------------------------------------------------------------
template <int B> Alina::SolverResult
block_solve(const Alina::PropertyTree& prm,
            size_t rows,
            std::vector<ptrdiff_t> const& ptr,
            std::vector<ptrdiff_t> const& col,
            std::vector<double> const& val,
            std::vector<double> const& rhs,
            std::vector<double>& x,
            bool reorder)
{
  auto& prof = Alina::Profiler::globalProfiler();

  typedef Alina::StaticMatrix<double, B, B> value_type;
  typedef Alina::StaticMatrix<double, B, 1> rhs_type;
  typedef Alina::BuiltinBackend<value_type> BBackend;

  typedef Alina::PreconditionedSolver<Alina::PreconditionerRuntime<BBackend>, Alina::SolverRuntime<BBackend>> Solver;

  auto As = std::tie(rows, ptr, col, val);
  auto Ab = Alina::adapter::block_matrix<value_type>(As);

  std::tuple<size_t, double> info;

  if (reorder) {
    prof.tic("reorder");
    Alina::adapter::reorder<> perm(Ab);
    prof.toc("reorder");

    prof.tic("setup");
    Solver solve(perm(Ab), prm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    rhs_type const* fptr = reinterpret_cast<rhs_type const*>(&rhs[0]);
    rhs_type* xptr = reinterpret_cast<rhs_type*>(&x[0]);

    Alina::numa_vector<rhs_type> F(perm(SmallSpan<const rhs_type>(fptr, rows / B)));
    Alina::numa_vector<rhs_type> X(perm(SmallSpan<rhs_type>(xptr, rows / B)));

    prof.tic("solve");
    info = solve(F, X);
    prof.toc("solve");

    perm.inverse(X, xptr);
  }
  else {
    prof.tic("setup");
    Solver solve(Ab, prm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    rhs_type const* fptr = reinterpret_cast<rhs_type const*>(&rhs[0]);
    rhs_type* xptr = reinterpret_cast<rhs_type*>(&x[0]);

    Alina::numa_vector<rhs_type> F(fptr, fptr + rows / B);
    Alina::numa_vector<rhs_type> X(xptr, xptr + rows / B);

    prof.tic("solve");
    info = solve(F, X);
    prof.toc("solve");

    std::copy(X.data(), X.data() + X.size(), xptr);
  }

  return info;
}
#endif

//---------------------------------------------------------------------------
Alina::SolverResult
scalar_solve(const Alina::PropertyTree& prm,
             size_t rows,
             std::vector<ptrdiff_t> const& ptr,
             std::vector<ptrdiff_t> const& col,
             std::vector<double> const& val,
             std::vector<double> const& rhs,
             std::vector<double>& x,
             bool reorder)
{
  std::cout << "Using scalar solve ptr_size=" << sizeof(ptrdiff_t)
            << " ptr_type_size=" << sizeof(Backend::ptr_type)
            << " col_type_size=" << sizeof(Backend::col_type)
            << " value_type_size=" << sizeof(Backend::value_type)
            << "\n";
  auto& prof = Alina::Profiler::globalProfiler();
  Backend::params bprm;

#if defined(SOLVER_BACKEND_CUDA)
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

  using Solver = Alina::PreconditionedSolver<Alina::PreconditionerRuntime<Backend>, Alina::SolverRuntime<Backend>>;

  Alina::SolverResult info;

  if (reorder) {
    prof.tic("reorder");
    Alina::adapter::reorder<> perm(std::tie(rows, ptr, col, val));
    prof.toc("reorder");

    prof.tic("setup");
    Solver solve(perm(std::tie(rows, ptr, col, val)), prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    std::vector<double> tmp(rows);

    perm.forward(rhs, tmp);
    auto f_b = Backend::copy_vector(tmp, bprm);

    perm.forward(x, tmp);
    auto x_b = Backend::copy_vector(tmp, bprm);

    prof.tic("solve");
    info = solve(*f_b, *x_b);
    prof.toc("solve");

#if defined(SOLVER_BACKEND_CUDA)
    thrust::copy(x_b->begin(), x_b->end(), tmp.begin());
#else
    std::copy(&(*x_b)[0], &(*x_b)[0] + rows, &tmp[0]);
#endif

    perm.inverse(tmp, x);
  }
  else {
    prof.tic("setup");
    Solver solve(std::tie(rows, ptr, col, val), prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    auto f_b = Backend::copy_vector(rhs, bprm);
    auto x_b = Backend::copy_vector(x, bprm);

    prof.tic("solve");
    info = solve(*f_b, *x_b);
    prof.toc("solve");

#if defined(SOLVER_BACKEND_CUDA)
    thrust::copy(x_b->begin(), x_b->end(), x.begin());
#else
    std::copy(&(*x_b)[0], &(*x_b)[0] + rows, &x[0]);
#endif
  }

  return info;
}

#define ARCCORE_ALINA_CALL_BLOCK_SOLVER(z, data, B)                                    \
  case B:                                                                      \
    return block_solve<B>(prm, rows, ptr, col, val, rhs, x, reorder);

//---------------------------------------------------------------------------
Alina::SolverResult
solve(const Alina::PropertyTree& prm,
      size_t rows,
      std::vector<ptrdiff_t> const& ptr,
      std::vector<ptrdiff_t> const& col,
      std::vector<double> const& val,
      std::vector<double> const& rhs,
      std::vector<double>& x,
      int block_size,
      bool reorder)
{
  switch (block_size) {
  case 1:
    return scalar_solve(prm, rows, ptr, col, val, rhs, x, reorder);
#if defined(SOLVER_BACKEND_BUILTIN)
    BOOST_PP_SEQ_FOR_EACH(ARCCORE_ALINA_CALL_BLOCK_SOLVER, ~, ARCCORE_ALINA_BLOCK_SIZES)
#endif
  default:
    precondition(false, "Unsupported block size");
    return {};
  }
}

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();
  namespace po = boost::program_options;
  namespace io = Alina::IO;

  using std::string;
  using std::vector;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Show this help.")("prm-file,P",
                                                  po::value<string>(),
                                                  "Parameter file in json format. ")(
  "prm,p",
  po::value<vector<string>>()->multitoken(),
  "Parameters specified as name=value pairs. "
  "May be provided multiple times. Examples:\n"
  "  -p solver.tol=1e-3\n"
  "  -p precond.coarse_enough=300")("matrix,A",
                                    po::value<string>(),
                                    "System matrix in the MatrixMarket format. "
                                    "When not specified, solves a Poisson problem in 3D unit cube. ")(
  "rhs,f",
  po::value<string>(),
  "The RHS vector in the MatrixMarket format. "
  "When omitted, a vector of ones is used by default. "
  "Should only be provided together with a system matrix. ")(
  "f0",
  po::bool_switch()->default_value(false),
  "Use zero RHS vector. Implies --random-initial and solver.ns_search=true")(
  "f1",
  po::bool_switch()->default_value(false),
  "Set RHS = Ax where x = 1")(
  "null,N",
  po::value<string>(),
  "The near null-space vectors in the MatrixMarket format. "
  "Should be a dense matrix of size N*M, where N is the number of "
  "unknowns, and M is the number of null-space vectors. "
  "Should only be provided together with a system matrix. ")(
  "coords,C",
  po::value<string>(),
  "Coordinate matrix where number of rows corresponds to the number of grid nodes "
  "and the number of columns corresponds to the problem dimensionality (2 or 3). "
  "Will be used to construct near null-space vectors as rigid body modes. "
  "Should only be provided together with a system matrix. ")(
  "binary,B",
  po::bool_switch()->default_value(false),
  "When specified, treat input files as binary instead of as MatrixMarket. "
  "It is assumed the files were converted to binary format with mm2bin utility. ")(
  "scale,s",
  po::bool_switch()->default_value(false),
  "Scale the matrix so that the diagonal is unit. ")(
  "block-size,b",
  po::value<int>()->default_value(1),
  "The block size of the system matrix. "
  "When specified, the system matrix is assumed to have block-wise structure. "
  "This usually is the case for problems in elasticity, structural mechanics, "
  "for coupled systems of PDE (such as Navier-Stokes equations), etc. ")(
  "size,n",
  po::value<int>()->default_value(32),
  "The size of the Poisson problem to solve when no system matrix is given. "
  "Specified as number of grid nodes along each dimension of a unit cube. "
  "The resulting system will have n*n*n unknowns. ")(
  "anisotropy,a",
  po::value<double>()->default_value(1.0),
  "The anisotropy value for the generated Poisson value. "
  "Used to determine problem scaling along X, Y, and Z axes: "
  "hy = hx * a, hz = hy * a.")(
  "single-level,1",
  po::bool_switch()->default_value(false),
  "When specified, the AMG hierarchy is not constructed. "
  "Instead, the problem is solved using a single-level smoother as preconditioner. ")(
  "reorder,r",
  po::bool_switch()->default_value(false),
  "When specified, the matrix will be reordered to improve cache-locality")(
  "initial,x",
  po::value<double>()->default_value(0),
  "Value to use as initial approximation. ")(
  "random-initial",
  po::bool_switch()->default_value(false),
  "Use random initial approximation. ")(
  "output,o",
  po::value<string>(),
  "Output file. Will be saved in the MatrixMarket format. "
  "When omitted, the solution is not saved. ");

  po::positional_options_description p;
  p.add("prm", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  for (int i = 0; i < argc; ++i) {
    if (i)
      std::cout << " ";
    std::cout << argv[i];
  }
  std::cout << std::endl;

  Alina::PropertyTree prm;
  if (vm.count("prm-file")) {
    prm.read_json(vm["prm-file"].as<string>());
  }

  if (vm.count("prm")) {
    for (const string& v : vm["prm"].as<vector<string>>()) {
      prm.putKeyValue(v);
    }
  }

  size_t rows, nv = 0;
  vector<ptrdiff_t> ptr, col;
  vector<double> val, rhs, null, x;

  if (vm.count("matrix")) {
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
    else if (vm["f1"].as<bool>()) {
      rhs.resize(rows);
      for (size_t i = 0; i < rows; ++i) {
        double s = 0;
        for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j)
          s += val[j];
        rhs[i] = s;
      }
    }
    else {
      rhs.resize(rows, vm["f0"].as<bool>() ? 0.0 : 1.0);
    }

    if (vm.count("null")) {
      string nfile = vm["null"].as<string>();

      size_t m;

      if (binary) {
        io::read_dense(nfile, m, nv, null);
      }
      else {
        std::tie(m, nv) = io::mm_reader(nfile)(null);
      }

      precondition(m == rows, "Near null-space vectors have wrong size");
    }
    else if (vm.count("coords")) {
      string cfile = vm["coords"].as<string>();
      std::vector<double> coo;

      size_t m, ndim;

      if (binary) {
        io::read_dense(cfile, m, ndim, coo);
      }
      else {
        std::tie(m, ndim) = io::mm_reader(cfile)(coo);
      }

      precondition(m * ndim == rows && (ndim == 2 || ndim == 3), "Coordinate matrix has wrong size");

      nv = Alina::rigid_body_modes(ndim, coo, null);
    }

    if (nv) {
      prm.put("precond.coarsening.nullspace.cols", nv);
      prm.put("precond.coarsening.nullspace.rows", rows);
      prm.put("precond.coarsening.nullspace.B", &null[0]);
    }
  }
  else {
    auto t = prof.scoped_tic("assembling");
    rows = sample_problem(vm["size"].as<int>(), val, col, ptr, rhs, vm["anisotropy"].as<double>());
  }

  if (vm["scale"].as<bool>()) {
    std::vector<double> dia(rows, 1.0);

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(rows); ++i) {
      double d = 1.0;
      for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
        if (col[j] == i) {
          d = 1 / sqrt(val[j]);
        }
      }
      if (!std::isnan(d))
        dia[i] = d;
    }

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(rows); ++i) {
      rhs[i] *= dia[i];
      for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
        val[j] *= dia[i] * dia[col[j]];
      }
    }
  }

  x.resize(rows, vm["initial"].as<double>());
  if (vm["random-initial"].as<bool>() || vm["f0"].as<bool>()) {
    std::mt19937 rng;
    std::uniform_real_distribution<double> rnd(-1, 1);
    for (auto& v : x)
      v = rnd(rng);
  }

  if (vm["f0"].as<bool>()) {
    prm.put("solver.ns_search", true);
  }

  int block_size = vm["block-size"].as<int>();
  std::cout << "BlockSize= " << block_size << "\n";

  if (vm["single-level"].as<bool>())
    prm.put("precond.class", "relaxation");

  String do_hypre_str = Platform::getEnvironmentVariable("ALINA_USE_HYPRE");
  bool do_hypre = false;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ALINA_USE_HYPRE", true))
    do_hypre = v.value();

  Alina::SolverResult solver_result;
#ifndef SOLVER_BACKEND_BUILTIN
  do_hypre = false;
#endif
  if (do_hypre) {
#ifdef SOLVER_BACKEND_BUILTIN
    _doHypreSolver(rows, ptr, col, val, rhs, x, argc, argv);
#endif
  }
  else {
    solver_result = solve(prm, rows, ptr, col, val, rhs, x, block_size, vm["reorder"].as<bool>());

    if (vm.count("output")) {
      auto t = prof.scoped_tic("write");
      Alina::IO::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }
  }
  std::cout << "Iterations: " << solver_result.nbIteration() << std::endl
            << "Error:      " << solver_result.residual() << std::endl
            << prof << std::endl;
}
