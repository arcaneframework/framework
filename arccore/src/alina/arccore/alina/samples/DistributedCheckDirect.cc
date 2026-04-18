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

#if defined(ARCCORE_ALINA_HAVE_EIGEN)
// To remove warnings about deprecated Eigen usage.
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#include "arccore/alina/DistributedEigenSparseLUDirectSolver.h"
#endif

#include "arccore/trace/ITraceMng.h"

#include "arccore/alina/DistributedDirectSolverRuntime.h"
#include "arccore/alina/Profiler.h"

#include "AlinaSamplesCommon.h"

#include <boost/program_options.hpp>

using namespace Arcane;

int main2(const Alina::SampleMainContext& ctx, int argc, char* argv[])
{
  ITraceMng* tm = ctx.traceMng();

  auto& prof = Alina::Profiler::globalProfiler();

  Alina::mpi_communicator comm(MPI_COMM_WORLD);

  tm->info() << "World size: " << comm.size;

  namespace po = boost::program_options;
  po::options_description desc("Options");

  desc.add_options()("help,h", "show help")("size,n",
                                            po::value<int>()->default_value(128),
                                            "domain size");
  desc.add_options()("prm-file,P",
                     po::value<std::string>(),
                     "Parameter file in json format. ");
  desc.add_options()("prm,p",
                     po::value<std::vector<std::string>>()->multitoken(),
                     "Parameters specified as name=value pairs. "
                     "May be provided multiple times. Examples:\n"
                     "  -p solver.tol=1e-3\n"
                     "  -p precond.coarse_enough=300");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
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

  const int n = vm["size"].as<int>();
  const int n2 = n * n;

  int chunk = (n2 + comm.size - 1) / comm.size;
  int chunk_start = comm.rank * chunk;
  int chunk_end = std::min(chunk_start + chunk, n2);

  chunk = chunk_end - chunk_start;

  std::vector<int> domain(comm.size + 1);
  MPI_Allgather(&chunk, 1, MPI_INT, &domain[1], 1, MPI_INT, comm);
  std::partial_sum(domain.begin(), domain.end(), domain.begin());

  prof.tic("assemble");
  Alina::CSRMatrix<double> A;
  A.set_size(chunk, domain.back(), true);
  A.set_nonzeros(chunk * 5);
  std::vector<double> rhs(chunk, 1);

  const double h2i = (n - 1) * (n - 1);
  for (int idx = chunk_start, row = 0, head = 0; idx < chunk_end; ++idx, ++row) {
    int j = idx / n;
    int i = idx % n;

    if (j > 0) {
      A.col[head] = idx - n;
      A.val[head] = -h2i;
      ++head;
    }

    if (i > 0) {
      A.col[head] = idx - 1;
      A.val[head] = -h2i;
      ++head;
    }

    A.col[head] = idx;
    A.val[head] = 4 * h2i;
    ++head;

    if (i + 1 < n) {
      A.col[head] = idx + 1;
      A.val[head] = -h2i;
      ++head;
    }

    if (j + 1 < n) {
      A.col[head] = idx + n;
      A.val[head] = -h2i;
      ++head;
    }

    A.ptr[row + 1] = head;
  }
  A.setNbNonZero(A.ptr[chunk]);
  prof.toc("assemble");

  std::vector<double> x(chunk);

  {
    auto t = prof.scoped_tic("skyline");

    prof.tic("setup");
    Alina::DistributedDirectSolverRuntime<double> solve(comm, A, prm);
    tm->info() << "Solver is = " << solve.type();
    prof.toc("setup");

    prof.tic("solve1");
    solve(rhs, x);
    prof.toc("solve1");
  }

#if defined(ARCCORE_ALINA_HAVE_EIGEN)
  {
    auto t = prof.scoped_tic("eigen");

    prof.tic("setup");
    Alina::DistributedEigenSparseLUDirectSolver<double> solve(comm, A, prm);
    prof.toc("setup");

    prof.tic("solve");
    std::vector<double> x2(chunk);
    solve(rhs, x2);
    prof.toc("solve");
    // Compare values between skyline and eigen
    Int32 nb_error = 0;
    for (Int32 i = 0; i < chunk; ++i) {
      Real x_ref = x[i];
      Real x_eigen = x2[i];
      Real abs_sum = std::abs(x_ref);

      Real diff = std::abs(x_eigen - x_ref);
      if (abs_sum != 0.0)
        diff /= abs_sum;

      if (diff > 1.0e-12) {
        tm->info() << "I=" << i << " compare skyline=" << x[i] << " eigen=" << x2[i] << " diff=" << diff;
        ++nb_error;
      }
    }
    if (nb_error != 0)
      ARCCORE_FATAL("Error when comparing Eigen and SkylineLU nb_error={0}", nb_error);
  }
#endif

  tm->info() << prof;
  return 0;
}

int main(int argc, char* argv[])
{
  return Arcane::Alina::SampleMainContext::execMain(main2, argc, argv);
}
