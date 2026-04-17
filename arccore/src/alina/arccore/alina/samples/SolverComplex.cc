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
#include <boost/range/iterator_range.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/ValueTypeComplex.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"

#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/PreconditionerRuntime.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/IO.h"

#include "arccore/alina/Profiler.h"

#include "SampleProblemCommon.h"

#ifndef ARCCORE_ALINA_BLOCK_SIZES
#define ARCCORE_ALINA_BLOCK_SIZES (2)(3)(4)
#endif
using namespace Arcane;
using namespace Arcane::Alina;

using Alina::precondition;

//---------------------------------------------------------------------------
template <class Precond, class Matrix>
Alina::SolverResult
solve(const Matrix& A,
      const Alina::PropertyTree& prm,
      std::vector<std::complex<double>> const& f,
      std::vector<std::complex<double>>& x)
{
  auto& prof = Alina::Profiler::globalProfiler();

  typedef typename Precond::backend_type Backend;

  typedef typename Alina::math::rhs_of<typename Backend::value_type>::type rhs_type;
  size_t n = Alina::backend::nbRow(A);

  rhs_type const* fptr = reinterpret_cast<rhs_type const*>(&f[0]);
  rhs_type* xptr = reinterpret_cast<rhs_type*>(&x[0]);
  SmallSpan<const rhs_type> frng(fptr, n);
  SmallSpan<rhs_type> xrng(xptr, n);

  using Solver = Alina::PreconditionedSolver<Precond, Alina::SolverRuntime<Backend>>;

  prof.tic("setup");
  Solver solve(A, prm);
  prof.toc("setup");

  std::cout << solve << std::endl;

  {
    auto t = prof.scoped_tic("solve");
    return solve(frng, xrng);
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
  "null,N",
  po::value<string>(),
  "The near null-space vectors in the MatrixMarket format. "
  "Should be a dense matrix of size N*M, where N is the number of "
  "unknowns, and M is the number of null-space vectors. "
  "Should only be provided together with a system matrix. ")(
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
  "size,n",
  po::value<int>()->default_value(32),
  "The size of the Poisson problem to solve when no system matrix is given. "
  "Specified as number of grid nodes along each dimension of a unit cube. "
  "The resulting system will have n*n*n unknowns. ")(
  "single-level,1",
  po::bool_switch()->default_value(false),
  "When specified, the AMG hierarchy is not constructed. "
  "Instead, the problem is solved using a single-level smoother as preconditioner. ")(
  "initial,x",
  po::value<double>()->default_value(0),
  "Value to use as initial approximation. ")(
  "output,o",
  po::value<string>(),
  "Output file. Will be saved in the MatrixMarket format. "
  "When omitted, the solution is not saved. ");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  Alina::PropertyTree prm;
  if (vm.count("prm-file")) {
    prm.read_json(vm["prm-file"].as<string>());
  }

  if (vm.count("prm")) {
    for (const string& v : vm["prm"].as<vector<string>>()) {
      prm.putKeyValue(v);
    }
  }

  size_t rows;
  vector<ptrdiff_t> ptr, col;
  vector<std::complex<double>> val, rhs, null, x;

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
    else {
      rhs.resize(rows, 1.0);
    }

    if (vm.count("null")) {
      string nfile = vm["null"].as<string>();

      size_t m, nv;

      if (binary) {
        io::read_dense(nfile, m, nv, null);
      }
      else {
        std::tie(m, nv) = io::mm_reader(nfile)(null);
      }

      precondition(m == rows, "Near null-space vectors have wrong size");

      prm.put("precond.coarsening.nullspace.cols", nv);
      prm.put("precond.coarsening.nullspace.rows", rows);
      prm.put("precond.coarsening.nullspace.B", &null[0]);
    }
  }
  else {
    auto t = prof.scoped_tic("assembling");
    rows = sample_problem(vm["size"].as<int>(), val, col, ptr, rhs);
  }

  x.resize(rows, vm["initial"].as<double>());

  if (vm["single-level"].as<bool>())
    prm.put("precond.class", "relaxation");

  int block_size = vm["block-size"].as<int>();
  Alina::SolverResult r;
#define CALL_BLOCK_SOLVER(z, data, B) \
  case B: { \
    typedef StaticMatrix<std::complex<double>, B, B> value_type; \
    typedef ::Arcane::Alina::BuiltinBackend<value_type> Backend; \
    r = solve<::Arcane::Alina::PreconditionerRuntime<Backend>>( \
    ::Arcane::Alina::adapter::block_matrix<value_type>( \
    std::tie(rows, ptr, col, val)), \
    prm, rhs, x); \
  } break;

  switch (block_size) {
  case 1: {
    typedef Alina::BuiltinBackend<std::complex<double>> Backend;
    r = solve<PreconditionerRuntime<Backend>>(
    std::tie(rows, ptr, col, val), prm, rhs, x);
  } break;
    BOOST_PP_SEQ_FOR_EACH(CALL_BLOCK_SOLVER, ~, ARCCORE_ALINA_BLOCK_SIZES)
  }

#undef CALL_BLOCK_SOLVER

  if (vm.count("output")) {
    auto t = prof.scoped_tic("write");
    Alina::IO::mm_write(vm["output"].as<string>(), &x[0], x.size());
  }

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl
            << prof << std::endl;
}
