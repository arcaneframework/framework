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

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/ValueTypeComplex.h"
#include "arccore/alina/Adapters.h"

#include "arccore/alina/DistributedPreconditionedSolver.h"
#include "arccore/alina/DistributedPreconditioner.h"
#include "arccore/alina/DistributedSolverRuntime.h"

#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

#include "arccore/common/internal/ProgramOptions.h"

#include "SampleProblemCommon.h"

using namespace Arcane;
using namespace Arcane::Alina;

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
  namespace po = Arcane::ProgramOptions;
  po::options_description desc("Options");

  desc.add_options()("help,h", "show help");
  desc.add_options()("size,n", po::value<ptrdiff_t>()->default_value(128), "domain size");
  desc.add_options()("prm-file,P", po::value<std::string>(), "Parameter file in json format. ");
  desc.add_options()("prm,p", po::value<std::vector<std::string>>()->multitoken(),
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
  n = sample_problem_distributed(comm.rank, comm.size, vm["size"].as<ptrdiff_t>(), 1, ptr, col, val, rhs);
  prof.toc("assemble");

  solve_scalar(comm, n, ptr, col, val, prm, rhs);
}
