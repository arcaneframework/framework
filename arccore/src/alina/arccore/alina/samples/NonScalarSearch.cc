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
  "scale,s",
  po::bool_switch()->default_value(false),
  "Scale the matrix so that the diagonal is unit. ")(
  "null,N",
  po::value<string>(),
  "Starting null-vectors in the MatrixMarket format. ")(
  "numvec,n",
  po::value<int>()->default_value(3),
  "The number of near nullspace vectors to search for. ")(
  "binary,B",
  po::bool_switch()->default_value(false),
  "When specified, treat input files as binary instead of as MatrixMarket. "
  "It is assumed the files were converted to binary format with mm2bin utility. ")(
  "output,o",
  po::value<string>(),
  "Output the computed nullspace to the MatrixMarket file.");

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

  ptrdiff_t rows, nv = 0, numvec = vm["numvec"].as<int>();
  vector<ptrdiff_t> ptr, col;
  vector<double> val, rhs;
  std::list<std::vector<double>> Z;

  {
    auto t = prof.scoped_tic("read");

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

    if (vm.count("null")) {
      string nfile = vm["null"].as<string>();

      std::vector<double> null;
      ptrdiff_t m;

      if (binary) {
        io::read_dense(nfile, m, nv, null);
      }
      else {
        std::tie(m, nv) = io::mm_reader(nfile)(null);
      }

      precondition(m == rows, "Near null-space vectors have wrong size");

      for (ptrdiff_t i = 0; i < nv; ++i) {
        Z.emplace_back(rows);
        for (ptrdiff_t j = 0; j < rows; ++j) {
          Z.back()[j] = null[j * nv + i];
        }
      }
    }
  }

  if (vm["scale"].as<bool>()) {
    auto t = prof.scoped_tic("scaling");
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

  using Backend = Alina::BuiltinBackend<double>;
  using Solver = Alina::PreconditionedSolver<Alina::AMG<Backend, Alina::CoarseningRuntime, Alina::RelaxationRuntime>,
                                             Alina::SolverRuntime<Backend>>;

  std::mt19937 rng;
  std::uniform_real_distribution<double> rnd(-1, 1);
  std::vector<double> x(rows), zero(rows, 0.0);

  auto A = std::tie(rows, ptr, col, val);

  prm.put("solver.ns_search", true);

  prof.tic("search");
  for (int k = nv; k < numvec; ++k) {
    auto t = prof.scoped_tic(std::string("vector ") + std::to_string(k));
    std::vector<double> N;

    if (k) {
      N.resize(k * rows);
      int j = 0;
      for (const auto& z : Z) {
        for (ptrdiff_t i = 0; i < rows; ++i) {
          N[i * k + j] = z[i];
        }
        ++j;
      }

      prm.put("precond.coarsening.nullspace.rows", rows);
      prm.put("precond.coarsening.nullspace.cols", k);
      prm.put("precond.coarsening.nullspace.B", N.data());
    }

    prof.tic("setup");
    Solver S(A, prm);
    prof.toc("setup");

    std::cout << std::endl
              << "-------------------------" << std::endl
              << "-- Searching for vector " << k << std::endl
              << "-------------------------" << std::endl
              << S << std::endl;

    for (auto& v : x)
      v = rnd(rng);

    prof.tic("solve");
    Alina::SolverResult r = S(zero, x);
    prof.toc("solve");

    std::cout << "Iterations: " << r.nbIteration() << std::endl
              << "Error:      " << r.residual() << std::endl;

    // Orthonormalize the new vector
    for (const auto& z : Z) {
      double c = Alina::backend::inner_product(x, z) / Alina::backend::inner_product(z, z);
      Alina::backend::axpby(-c, z, 1, x);
    }

    double nx = sqrt(Alina::backend::inner_product(x, x));
    for (auto& v : x)
      v /= nx;
    Z.push_back(x);
  }
  prof.toc("search");

  // Solve the system using the near nullspace vectors:
  std::vector<double> N(numvec * rows);
  {
    auto t = prof.scoped_tic("apply");

    int j = 0;
    for (const auto& z : Z) {
      for (ptrdiff_t i = 0; i < rows; ++i) {
        N[i * numvec + j] = z[i];
      }
      ++j;
    }

    prm.put("precond.coarsening.nullspace.rows", rows);
    prm.put("precond.coarsening.nullspace.cols", numvec);
    prm.put("precond.coarsening.nullspace.B", N.data());

    prof.tic("setup");
    Solver S(A, prm);
    prof.toc("setup");

    std::cout << std::endl
              << "-------------------------" << std::endl
              << "-- Solving the system " << std::endl
              << "-------------------------" << std::endl
              << S << std::endl;

    Alina::backend::clear(x);

    prof.tic("solve");
    Alina::SolverResult r = S(rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << r.nbIteration() << std::endl
              << "Error:      " << r.residual() << std::endl;
  }

  if (vm.count("output")) {
    auto t = prof.scoped_tic("write");
    Alina::IO::mm_write(vm["output"].as<string>(), N.data(), rows, numvec);
  }

  std::cout << prof << std::endl;
}
