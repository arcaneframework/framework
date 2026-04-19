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
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>

// To remove warnings about deprecated Eigen usage.
//#pragma GCC diagnostic ignored "-Wdeprecated-copy"
//ragma GCC diagnostic ignored "-Wint-in-bool-context"

#if defined(SOLVER_BACKEND_CUDA)
// This seems not defined with CUDA
namespace boost::math
{
class rounding_error
{};
} // namespace boost::math
#endif

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>

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

#include "arccore/trace/ITraceMng.h"

#include "arccore/alina/DistributedDirectSolverRuntime.h"
#include "arccore/alina/DistributedSolverRuntime.h"
#include "arccore/alina/DistributedSubDomainDeflation.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/Profiler.h"
#include "AlinaSamplesCommon.h"

using namespace Arcane;
using namespace Arcane::Alina;

#include "DomainPartition.h"

struct deflation_vectors
{
  size_t nv;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  deflation_vectors(ptrdiff_t n, size_t nv = 4)
  : nv(nv)
  , x(n)
  , y(n)
  , z(n)
  {}

  size_t dim() const { return nv; }

  double operator()(ptrdiff_t i, int j) const
  {
    switch (j) {
    default:
    case 0:
      return 1;
    case 1:
      return x[i];
    case 2:
      return y[i];
    case 3:
      return z[i];
    }
  }
};

struct renumbering
{
  const DomainPartition<3>& part;
  const std::vector<ptrdiff_t>& dom;

  renumbering(const DomainPartition<3>& p,
              const std::vector<ptrdiff_t>& d)
  : part(p)
  , dom(d)
  {}

  ptrdiff_t operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const
  {
    boost::array<ptrdiff_t, 3> p = { { i, j, k } };
    std::pair<int, ptrdiff_t> v = part.index(p);
    return dom[v.first] + v.second;
  }
};

int main2(const Alina::SampleMainContext& ctx, int argc, char* argv[])
{
  auto& prof = Alina::Profiler::globalProfiler();
  ITraceMng* tm = ctx.traceMng();
  Alina::mpi_communicator world(MPI_COMM_WORLD);

  tm->info() << "World size: " << world.size;

  // Read configuration from command line
  ptrdiff_t n = 128;
  bool constant_deflation = false;

  auto coarsening = Alina::eCoarserningType::smoothed_aggregation;
  auto relaxation = Alina::eRelaxationType::spai0;
  auto iterative_solver = Alina::eSolverType::bicgstabl;
  auto direct_solver = Alina::eDistributedDirectSolverType::skyline_lu;

  bool just_relax = false;
  bool symm_dirichlet = true;
  std::string parameter_file;

  namespace po = boost::program_options;
  po::options_description desc("Options");

  desc.add_options()("help,h", "show help")(
  "symbc",
  po::value<bool>(&symm_dirichlet)->default_value(symm_dirichlet),
  "Use symmetric Dirichlet conditions in laplace2d")(
  "size,n",
  po::value<ptrdiff_t>(&n)->default_value(n),
  "domain size")(
  "coarsening,c",
  po::value<Alina::eCoarserningType>(&coarsening)->default_value(coarsening),
  "ruge_stuben, aggregation, smoothed_aggregation, smoothed_aggr_emin")(
  "relaxation,r",
  po::value<Alina::eRelaxationType>(&relaxation)->default_value(relaxation),
  "gauss_seidel, ilu0, iluk, ilut, damped_jacobi, spai0, spai1, chebyshev")(
  "iter_solver,i",
  po::value<Alina::eSolverType>(&iterative_solver)->default_value(iterative_solver),
  "cg, bicgstab, bicgstabl, gmres")(
  "dir_solver,d",
  po::value<Alina::eDistributedDirectSolverType>(&direct_solver)->default_value(direct_solver),
  "skyline_lu"
#ifdef ARCCORE_ALINA_HAVE_EIGEN
  ", eigen_splu"
#endif
  )(
  "cd",
  po::bool_switch(&constant_deflation),
  "Use constant deflation (linear deflation is used by default)")(
  "params,P",
  po::value<std::string>(&parameter_file),
  "parameter file in json format")(
  "prm,p",
  po::value<std::vector<std::string>>()->multitoken(),
  "Parameters specified as name=value pairs. "
  "May be provided multiple times. Examples:\n"
  "  -p solver.tol=1e-3\n"
  "  -p precond.coarse_enough=300")(
  "just-relax,0",
  po::bool_switch(&just_relax),
  "Do not create AMG hierarchy, use relaxation as preconditioner");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  Alina::PropertyTree prm;
  if (vm.count("params"))
    prm.read_json(parameter_file);

  if (vm.count("prm")) {
    for (const std::string& v : vm["prm"].as<std::vector<std::string>>()) {
      prm.putKeyValue(v);
    }
  }

  prm.put("isolver.type", iterative_solver);
  prm.put("dsolver.type", direct_solver);

  boost::array<ptrdiff_t, 3> lo = { { 0, 0, 0 } };
  boost::array<ptrdiff_t, 3> hi = { { n - 1, n - 1, n - 1 } };

  prof.tic("partition");
  DomainPartition<3> part(lo, hi, world.size);
  ptrdiff_t chunk = part.size(world.rank);

  std::vector<ptrdiff_t> domain(world.size + 1);
  MPI_Allgather(&chunk, 1, Alina::mpi_datatype<ptrdiff_t>(),
                &domain[1], 1, Alina::mpi_datatype<ptrdiff_t>(), world);
  std::partial_sum(domain.begin(), domain.end(), domain.begin());

  lo = part.domain(world.rank).min_corner();
  hi = part.domain(world.rank).max_corner();

  renumbering renum(part, domain);

  deflation_vectors def(chunk, constant_deflation ? 1 : 4);
  for (ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
    for (ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
      for (ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
        boost::array<ptrdiff_t, 3> p = { { i, j, k } };
        std::pair<int, ptrdiff_t> v = part.index(p);

        def.x[v.second] = (i - (lo[0] + hi[0]) / 2);
        def.y[v.second] = (j - (lo[1] + hi[1]) / 2);
        def.z[v.second] = (k - (lo[2] + hi[2]) / 2);
      }
    }
  }
  prof.toc("partition");

  prof.tic("assemble");
  std::vector<ptrdiff_t> ptr;
  std::vector<ptrdiff_t> col;
  std::vector<double> val;
  std::vector<double> rhs;

  ptr.reserve(chunk + 1);
  col.reserve(chunk * 7);
  val.reserve(chunk * 7);
  rhs.reserve(chunk);

  ptr.push_back(0);

  const double h2i = (n - 1) * (n - 1);

  for (ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
    for (ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
      for (ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {

        if (!symm_dirichlet && (i == 0 || j == 0 || k == 0 || i + 1 == n || j + 1 == n || k + 1 == n)) {
          col.push_back(renum(i, j, k));
          val.push_back(1);
          rhs.push_back(0);
        }
        else {
          if (k > 0) {
            col.push_back(renum(i, j, k - 1));
            val.push_back(-h2i);
          }

          if (j > 0) {
            col.push_back(renum(i, j - 1, k));
            val.push_back(-h2i);
          }

          if (i > 0) {
            col.push_back(renum(i - 1, j, k));
            val.push_back(-h2i);
          }

          col.push_back(renum(i, j, k));
          val.push_back(6 * h2i);

          if (i + 1 < n) {
            col.push_back(renum(i + 1, j, k));
            val.push_back(-h2i);
          }

          if (j + 1 < n) {
            col.push_back(renum(i, j + 1, k));
            val.push_back(-h2i);
          }

          if (k + 1 < n) {
            col.push_back(renum(i, j, k + 1));
            val.push_back(-h2i);
          }

          rhs.push_back(1);
        }
        ptr.push_back(col.size());
      }
    }
  }
  prof.toc("assemble");

  Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
  vex::Context ctx(vex::Filter::Env);
  std::cout << ctx << std::endl;
  bprm.q = ctx;
#elif defined(SOLVER_BACKEND_CUDA)
  cusparseCreate(&bprm.cusparse_handle);
#endif

  auto f = Backend::copy_vector(rhs, bprm);
  auto x = Backend::create_vector(chunk, bprm);

  Alina::backend::clear(*x);

  size_t iters;
  double resid;

  std::function<double(ptrdiff_t, unsigned)> def_vec = std::cref(def);
  prm.put("num_def_vec", def.dim());
  prm.put("def_vec", &def_vec);

  if (just_relax) {
    prm.put("local.type", relaxation);

    prof.tic("setup");
    using SDD = DistributedSubDomainDeflation<RelaxationAsPreconditioner<Backend, RelaxationRuntime>,
                                              DistributedSolverRuntime<Backend>,
                                              DistributedDirectSolverRuntime<double>>;

    SDD solve(world, std::tie(chunk, ptr, col, val), prm, bprm);
    prof.toc("setup");

    prof.tic("solve");
    std::tie(iters, resid) = solve(*f, *x);
    prof.toc("solve");
  }
  else {
    prm.put("local.coarsening.type", coarsening);
    prm.put("local.relax.type", relaxation);

    prof.tic("setup");
    using SDD = DistributedSubDomainDeflation<AMG<Backend, CoarseningRuntime, RelaxationRuntime>,
                                              DistributedSolverRuntime<Backend>,
                                              DistributedDirectSolverRuntime<double>>;

    SDD solve(world, std::tie(chunk, ptr, col, val), prm, bprm);
    prof.toc("setup");

    prof.tic("solve");
    std::tie(iters, resid) = solve(*f, *x);
    prof.toc("solve");
  }

  tm->info() << "Iterations: " << iters << "\n"
             << "Error:      " << resid << "\n\n"
             << prof << "\n";
  return 0;
}

int main(int argc, char* argv[])
{
  return Arcane::Alina::SampleMainContext::execMain(main2, argc, argv);
}
