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

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/PreconditionerRuntime.h"
#include "arccore/alina/DeflatedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/IO.h"

#include "arccore/alina/Profiler.h"

using namespace Arcane;

using Alina::precondition;

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
                                    po::value<string>()->required(),
                                    "System matrix in the MatrixMarket format.")(
  "rhs,f",
  po::value<string>(),
  "The RHS vector in the MatrixMarket format. "
  "When omitted, a vector of ones is used by default. "
  "Should only be provided together with a system matrix. ")(
  "defvec,D",
  po::value<string>(),
  "The near null-space vectors in the MatrixMarket format. ")(
  "coords,C",
  po::value<string>(),
  "Coordinate matrix where number of rows corresponds to the number of grid nodes "
  "and the number of columns corresponds to the problem dimensionality (2 or 3). "
  "Will be used to construct near null-space vectors as rigid body modes. ")(
  "binary,B",
  po::bool_switch()->default_value(false),
  "When specified, treat input files as binary instead of as MatrixMarket. "
  "It is assumed the files were converted to binary format with mm2bin utility. ")(
  "single-level,1",
  po::bool_switch()->default_value(false),
  "When specified, the AMG hierarchy is not constructed. "
  "Instead, the problem is solved using a single-level smoother as preconditioner. ")(
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

  if (!vm.count("defvec") && !vm.count("coords")) {
    std::cerr << "Either defvec or coords should be given" << std::endl;
    return 1;
  }

  ptrdiff_t rows, nv;
  vector<ptrdiff_t> ptr, col;
  vector<double> val, rhs, z;

  {
    auto t = prof.scoped_tic("reading");

    string Afile = vm["matrix"].as<string>();
    bool binary = vm["binary"].as<bool>();

    if (binary) {
      io::read_crs(Afile, rows, ptr, col, val);
    }
    else {
      ptrdiff_t cols;
      std::tie(rows, cols) = io::mm_reader(Afile)(ptr, col, val);
      precondition(rows == cols, "Non-square system matrix");
    }

    if (vm.count("rhs")) {
      string bfile = vm["rhs"].as<string>();

      ptrdiff_t n, m;

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

    if (vm.count("defvec")) {
      string nfile = vm["defvec"].as<string>();
      std::vector<double> N;

      ptrdiff_t m;

      if (binary) {
        io::read_dense(nfile, m, nv, N);
      }
      else {
        std::tie(m, nv) = io::mm_reader(nfile)(N);
      }

      precondition(m == rows, "Deflation vectors have wrong size");

      z.resize(N.size());
      for (ptrdiff_t i = 0; i < rows; ++i)
        for (ptrdiff_t j = 0; j < nv; ++j)
          z[i + j * rows] = N[i * nv + j];
    }
    else if (vm.count("coords")) {
      string cfile = vm["coords"].as<string>();
      std::vector<double> coo;

      ptrdiff_t m, ndim;

      if (binary) {
        io::read_dense(cfile, m, ndim, coo);
      }
      else {
        std::tie(m, ndim) = io::mm_reader(cfile)(coo);
      }

      precondition(m * ndim == rows && (ndim == 2 || ndim == 3), "Coordinate matrix has wrong size");

      nv = Alina::rigid_body_modes(ndim, coo, z, /*transpose = */ true);
    }

    prm.put("nvec", nv);
    prm.put("vec", z.data());
  }

  std::vector<double> x(rows, 0);

  if (vm["single-level"].as<bool>())
    prm.put("precond.class", "relaxation");

  typedef Alina::BuiltinBackend<double> Backend;
  typedef Alina::DeflatedSolver<Alina::PreconditionerRuntime<Backend>,
                                Alina::SolverRuntime<Backend>>
  Solver;

  auto A = std::tie(rows, ptr, col, val);

  prof.tic("setup");
  Solver solve(A, prm);
  prof.toc("setup");

  prof.tic("solve");
  Alina::SolverResult result = solve(rhs, x);
  prof.toc("solve");

  if (vm.count("output")) {
    auto t = prof.scoped_tic("write");
    Alina::IO::mm_write(vm["output"].as<string>(), x.data(), x.size());
  }

  std::vector<double> r(rows);
  Alina::backend::residual(rhs, A, x, r);

  std::cout << "Iterations: " << result.nbIteration() << std::endl
            << "Error:      " << result.residual() << std::endl
            << "True error: " << sqrt(Alina::backend::inner_product(r, r)) / sqrt(Alina::backend::inner_product(rhs, rhs))
            << prof << std::endl;
}
