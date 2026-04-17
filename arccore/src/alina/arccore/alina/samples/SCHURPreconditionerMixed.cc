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
#include <string>

#include <boost/program_options.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/make_block_solver.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/SchurPressureCorrectionPreconditioner.h"
#include "arccore/alina/PreconditionerRuntime.h"
#include "arccore/alina/Adapters.h"

#if defined(SOLVER_BACKEND_VEXCL)
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#include "arccore/alina/BuiltinBackend.h"
#ifdef BLOCK_TYPE_EIGEN
#include "arccore/alina/ValueTypeEigen.h"
template <class T, int N, int M>
using BlockMatrix = Eigen::Matrix<T, N, M>;
#  else
     template <class T, int N, int M>
     using BlockMatrix = Arcane::Alina::StaticMatrix<T, N, M>;
#endif
template <class T> using Backend = Arcane::Alina::BuiltinBackend<T>;
#endif

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

#ifndef ARCCORE_ALINA_BLOCK_SIZES
#  define ARCCORE_ALINA_BLOCK_SIZES (3)(4)
#endif

using namespace Arcane;

using Alina::precondition;

//---------------------------------------------------------------------------
template <class USolver, class PSolver, class Matrix>
void solve_schur(const Matrix& K, const std::vector<double>& rhs, Alina::PropertyTree& prm)
{
  auto& prof = Alina::Profiler::globalProfiler();
  typedef Backend<double> SBackend;
  SBackend::params bprm;

  auto t1 = prof.scoped_tic("schur_complement");

  prof.tic("setup");
  Alina::PreconditionedSolver<Alina::preconditioner::SchurPressureCorrectionPreconditioner<USolver, PSolver>,
                              Alina::SolverRuntime<SBackend>>
  solve(K, prm, bprm);
  prof.toc("setup");

  std::cout << solve << std::endl;

  auto A = SBackend::copy_matrix(std::make_shared<Alina::CSRMatrix<double>>(K), bprm);
  auto f = SBackend::copy_vector(rhs, bprm);
  auto x = SBackend::create_vector(rhs.size(), bprm);
  Alina::backend::clear(*x);

  prof.tic("solve");
  Alina::SolverResult r = solve(*A, *f, *x);
  prof.toc("solve");

  std::cout << "Iterations: " << r.nbIteration() << std::endl
            << "Error:      " << r.residual() << std::endl;
}

#define ARCCORE_ALINA_BLOCK_PSOLVER(z, data, B)                 \
  case B: {                                             \
    typedef Backend<BlockMatrix<float, B, B>> BBackend; \
    typedef ::Arcane::Alina::make_block_solver<                   \
        ::Arcane::Alina::PreconditionerRuntime<BBackend>,       \
        ::Arcane::Alina::SolverRuntime<BBackend> >     \
        PSolver;                                        \
    solve_schur<USolver, PSolver>(K, rhs, prm);         \
  } break;

//---------------------------------------------------------------------------
template <class USolver, class Matrix>
void solve_schur(int pb, const Matrix& K, const std::vector<double>& rhs, Alina::PropertyTree& prm)
{
  switch (pb) {
  case 1: {
    typedef Alina::PreconditionedSolver<
    Alina::PreconditionerRuntime<Backend<float>>,
    Alina::SolverRuntime<Backend<float>>>
    PSolver;
    solve_schur<USolver, PSolver>(K, rhs, prm);
  } break;
#if defined(SOLVER_BACKEND_BUILTIN) || defined(SOLVER_BACKEND_VEXCL)
    BOOST_PP_SEQ_FOR_EACH(ARCCORE_ALINA_BLOCK_PSOLVER, ~, ARCCORE_ALINA_BLOCK_SIZES)
#endif
  default:
    precondition(false, "Unsupported block size for pressure");
  }
}

#define ARCCORE_ALINA_BLOCK_USOLVER(z, data, B) \
  case B: { \
    typedef Backend<BlockMatrix<float, B, B>> BBackend; \
    typedef ::Arcane::Alina::make_block_solver< \
    ::Arcane::Alina::PreconditionerRuntime<BBackend>, \
    ::Arcane::Alina::SolverRuntime<BBackend>> \
    USolver; \
    solve_schur<USolver>(pb, K, rhs, prm); \
  } break;

//---------------------------------------------------------------------------
template <class Matrix>
void solve_schur(int ub, int pb, const Matrix& K, const std::vector<double>& rhs, Alina::PropertyTree& prm)
{
  switch (ub) {
  case 1: {
    using USolver = Alina::PreconditionedSolver<Alina::PreconditionerRuntime<Backend<float>>,
                                                Alina::SolverRuntime<Backend<float>>>;
    solve_schur<USolver>(pb, K, rhs, prm);
  } break;
#if defined(SOLVER_BACKEND_BUILTIN) || defined(SOLVER_BACKEND_VEXCL)
    BOOST_PP_SEQ_FOR_EACH(ARCCORE_ALINA_BLOCK_USOLVER, ~, ARCCORE_ALINA_BLOCK_SIZES)
#endif
  default:
    precondition(false, "Unsupported block size for flow");
  }
}

//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();
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
  "scale,s",
  po::bool_switch()->default_value(false),
  "Scale the matrix so that the diagonal is unit. ")(
  "matrix,A",
  po::value<string>()->required(),
  "The system matrix in MatrixMarket format")(
  "rhs,f",
  po::value<string>(),
  "The right-hand side in MatrixMarket format")(
  "pmask,m",
  po::value<string>(),
  "The pressure mask in MatrixMarket format. Or, if the parameter has "
  "the form '%n:m', then each (n+i*m)-th variable is treated as pressure.")(
  "ub",
  po::value<int>()->default_value(1),
  "Block-size of the 'flow'/'non-pressure' part of the matrix")(
  "pb",
  po::value<int>()->default_value(1),
  "Block-size of the 'pressure' part of the matrix")(
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

  size_t rows;
  vector<ptrdiff_t> ptr, col;
  vector<double> val, rhs;
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

    if (vm.count("pmask")) {
      std::string pmask = vm["pmask"].as<string>();
      prm.put("precond.pmask_size", rows);

      switch (pmask[0]) {
      case '%':
      case '<':
      case '>':
        prm.put("precond.pmask_pattern", pmask);
        break;
      default: {
        size_t n, m;
        std::tie(n, m) = Alina::IO::mm_reader(pmask)(pm);
        precondition(n == rows && m == 1, "Mask file has wrong size");
        prm.put("precond.pmask", static_cast<void*>(&pm[0]));
      }
      }
    }
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

  solve_schur(vm["ub"].as<int>(), vm["pb"].as<int>(),
              std::tie(rows, ptr, col, val), rhs, prm);

  std::cout << prof << std::endl;
}
