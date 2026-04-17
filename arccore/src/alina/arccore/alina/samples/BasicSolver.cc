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
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Convert.h"

#include "arccore/alina/Relaxation.h"
#include "arccore/alina/Coarsening.h"
#include "arccore/alina/BiCGStabSolver.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/IO.h"

#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/PreconditionerRuntime.h"

#include "arccore/alina/Profiler.h"

#include "AlinaSamplesCommon.h"
#include "arccore/trace/ITraceMng.h"

#include "SampleProblemCommon.h"

using namespace Arcane;
using Alina::precondition;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Use 32 bit indexing for backend.
using Backend = Arcane::Alina::BuiltinBackend<double, Arcane::Int32>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_doHypreSolver(int nb_row,
               std::vector<ptrdiff_t> const& ptr,
               std::vector<ptrdiff_t> const& col,
               std::vector<double> const& val,
               std::vector<double> const& rhs,
               std::vector<double>& x,
               int argc, char* argv[]);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Alina::SolverResult
solve(const Alina::PropertyTree& prm,
      size_t rows,
      std::vector<ptrdiff_t> const& ptr,
      std::vector<ptrdiff_t> const& col,
      std::vector<double> const& val,
      std::vector<double> const& rhs,
      std::vector<double>& x)
{
  std::cout << "Using scalar solve ptr_size=" << sizeof(ptrdiff_t)
            << " ptr_type_size=" << sizeof(Backend::ptr_type)
            << " col_type_size=" << sizeof(Backend::col_type)
            << " value_type_size=" << sizeof(Backend::value_type)
            << "\n";
  auto& prof = Alina::Profiler::globalProfiler();
  Backend::params bprm;

  using Precond = Alina::AMG<Backend, Alina::AggregationCoarsening, Alina::SPAI0Relaxation>;
  //using Solver = Alina::PreconditionedSolver<Precond, Alina::BiCGStabSolver<Backend>>;
  using Solver = Alina::PreconditionedSolver<Precond, Alina::SolverRuntime<Backend>>;
  //using Solver = Alina::PreconditionedSolver<Alina::PreconditionerRuntime<Backend>, Alina::SolverRuntime<Backend>>;
  //using Solver = Alina::PreconditionedSolver<Alina::PreconditionerRuntime<Backend>, Alina::BiCGStabSolver<Backend>>;

  Alina::SolverResult info;

  {
    prof.tic("setup");
    Solver solve(std::tie(rows, ptr, col, val), prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    auto f_b = Backend::copy_vector(rhs, bprm);
    auto x_b = Backend::copy_vector(x, bprm);

    prof.tic("solve");
    info = solve(*f_b, *x_b);
    prof.toc("solve");
  }

  return info;
}

//---------------------------------------------------------------------------

int main2(const Alina::SampleMainContext& ctx, int argc, char* argv[])
{
  ITraceMng* tm = ctx.traceMng();

  auto& prof = Alina::Profiler::globalProfiler();
  namespace po = boost::program_options;
  namespace io = Alina::IO;

  using std::string;
  using std::vector;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Show this help.");
  desc.add_options()("prm-file,P", po::value<string>(),
                     "Parameter file in json format. ");
  desc.add_options()("prm,p", po::value<vector<string>>()->multitoken(),
                     "Parameters specified as name=value pairs. "
                     "May be provided multiple times. Examples:\n"
                     "  -p solver.tol=1e-3\n"
                     "  -p precond.coarse_enough=300");

  desc.add_options()("size,n",
                     po::value<int>()->default_value(32),
                     "The size of the Poisson problem to solve when no system matrix is given. "
                     "Specified as number of grid nodes along each dimension of a unit cube. "
                     "The resulting system will have n*n*n unknowns. ");

  desc.add_options()("anisotropy,a",
                     po::value<double>()->default_value(1.0),
                     "The anisotropy value for the generated Poisson value. "
                     "Used to determine problem scaling along X, Y, and Z axes: "
                     "hy = hx * a, hz = hy * a.");

  po::positional_options_description p;
  p.add("prm", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    tm->info() << desc;
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

  size_t rows = vm["size"].as<int>();
  vector<ptrdiff_t> ptr, col;
  vector<double> val, rhs, null, x;
  std::cout << "ROWS=" << rows << "\n";
  {
    auto t = prof.scoped_tic("assembling");
    rows = sample_problem(rows, val, col, ptr, rhs, vm["anisotropy"].as<double>());
  }

  x.resize(rows, 0.0);

  String do_hypre_str = Platform::getEnvironmentVariable("ALINA_USE_HYPRE");
  bool do_hypre = false;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ALINA_USE_HYPRE", true))
    do_hypre = v.value();

  Alina::SolverResult solver_result;
  if (do_hypre) {
    _doHypreSolver(rows, ptr, col, val, rhs, x, argc, argv);
  }
  else {
    solver_result = solve(prm, rows, ptr, col, val, rhs, x);

    if (vm.count("output")) {
      auto t = prof.scoped_tic("write");
      Alina::IO::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }
  }
  std::cout << "Iterations: " << solver_result.nbIteration() << std::endl
            << "Error:      " << solver_result.residual() << std::endl
            << prof << std::endl;
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
