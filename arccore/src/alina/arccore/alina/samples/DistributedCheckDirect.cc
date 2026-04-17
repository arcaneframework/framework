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
#include <algorithm>
#include <numeric>
#include <cmath>

// To remove warnings about deprecated Eigen usage.
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>

#include "arccore/alina/DistributedDirectSolverRuntime.h"
#include "arccore/alina/Profiler.h"

namespace Arcane::Alina
{
Profiler prof;
}

using namespace Arcane;

int main(int argc, char* argv[])
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  BOOST_SCOPE_EXIT(void)
  {
    MPI_Finalize();
  }
  BOOST_SCOPE_EXIT_END

  Alina::mpi_communicator comm(MPI_COMM_WORLD);

  if (comm.rank == 0)
    std::cout << "World size: " << comm.size << std::endl;

  namespace po = boost::program_options;
  po::options_description desc("Options");

  desc.add_options()("help,h", "show help")(
  "size,n",
  po::value<int>()->default_value(128),
  "domain size")("prm-file,P",
                 po::value<std::string>(),
                 "Parameter file in json format. ")(
  "prm,p",
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

  using Alina::prof;

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

  prof.tic("setup");
  Alina::DistributedDirectSolverRuntime<double> solve(comm, A, prm);
  prof.toc("setup");

  prof.tic("solve");
  std::vector<double> x(chunk);
  solve(rhs, x);
  solve(rhs, x);
  prof.toc("solve");

  prof.tic("save");
  if (comm.rank == 0) {
    std::vector<double> X(n2);
    std::copy(x.begin(), x.end(), X.begin());

    for (int i = 1; i < comm.size; ++i)
      MPI_Recv(&X[domain[i]], domain[i + 1] - domain[i], MPI_DOUBLE, i, 42, comm, MPI_STATUS_IGNORE);

    std::ofstream f("out.dat", std::ios::binary);
    f.write((char*)&n2, sizeof(int));
    f.write((char*)X.data(), n2 * sizeof(double));
  }
  else {
    MPI_Send(x.data(), chunk, MPI_DOUBLE, 0, 42, comm);
  }
  prof.toc("save");

  if (comm.rank == 0) {
    std::cout << prof << std::endl;
  }
}
