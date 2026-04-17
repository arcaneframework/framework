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
#include <complex>

#include <boost/program_options.hpp>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/ValueTypeComplex.h"
#include "arccore/alina/Adapters.h"

#include "arccore/alina/DistributedPreconditionedSolver.h"
#include "arccore/alina/DistributedPreconditioner.h"
#include "arccore/alina/DistributedSolverRuntime.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;
using namespace Arcane::Alina;

namespace math = Alina::math;

//---------------------------------------------------------------------------
ptrdiff_t
assemble_poisson3d(Alina::mpi_communicator comm,
                   ptrdiff_t n, int block_size,
                   std::vector<ptrdiff_t>& ptr,
                   std::vector<ptrdiff_t>& col,
                   std::vector<std::complex<double>>& val,
                   std::vector<std::complex<double>>& rhs)
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
                  const std::vector<std::complex<double>>& val,
                  const Alina::PropertyTree& prm,
                  const std::vector<std::complex<double>>& rhs)
{
  auto& prof = Alina::Profiler::globalProfiler();
  using Backend = Alina::BuiltinBackend<std::complex<double>>;

  using Solver = Alina::DistributedPreconditionedSolver<Alina::DistributedPreconditioner<Backend>,
                                                        Alina::DistributedSolverRuntime<Backend>>;

  prof.tic("setup");
  Solver solve(comm, std::tie(chunk, ptr, col, val), prm);
  prof.toc("setup");

  if (comm.rank == 0) {
    std::cout << solve << std::endl;
  }

  std::vector<std::complex<double>> x(chunk);

  prof.tic("solve");
  Alina::SolverResult r = solve(rhs, x);
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

  desc.add_options()("help,h", "show help")(
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
  "  -p precond.coarse_enough=300");

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
  std::vector<std::complex<double>> val;
  std::vector<std::complex<double>> rhs;

  prof.tic("assemble");
  n = assemble_poisson3d(comm, vm["size"].as<ptrdiff_t>(), 1, ptr, col, val, rhs);
  prof.toc("assemble");

  solve_scalar(comm, n, ptr, col, val, prm, rhs);
}
