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
#include <numeric>

#include <boost/scope_exit.hpp>

#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/AlinaLib.h"
#include "AlinaSamplesCommon.h"
#include "arccore/trace/ITraceMng.h"

#include "DomainPartition.h"

double constant_deflation(int, ptrdiff_t, void*)
{
  return 1;
}

using namespace Arcane;

int main2(const Alina::SampleMainContext& ctx, int argc, char* argv[])
{
  ITraceMng* tm = ctx.traceMng();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
    tm->info() << "World size: " << size;

  const ptrdiff_t n = argc > 1 ? atoi(argv[1]) : 1024;
  const ptrdiff_t n2 = n * n;

  // Partition
  boost::array<ptrdiff_t, 2> lo = { { 0, 0 } };
  boost::array<ptrdiff_t, 2> hi = { { n - 1, n - 1 } };

  DomainPartition<2> part(lo, hi, size);
  ptrdiff_t chunk = part.size(rank);

  std::vector<ptrdiff_t> domain(size + 1);
  MPI_Allgather(&chunk, 1, Alina::mpi_datatype<ptrdiff_t>(), &domain[1], 1, Alina::mpi_datatype<ptrdiff_t>(), MPI_COMM_WORLD);
  std::partial_sum(domain.begin(), domain.end(), domain.begin());

  ptrdiff_t chunk_start = domain[rank];
  ptrdiff_t chunk_end = domain[rank + 1];

  std::vector<ptrdiff_t> renum(n2);
  for (ptrdiff_t j = 0, idx = 0; j < n; ++j) {
    for (ptrdiff_t i = 0; i < n; ++i, ++idx) {
      boost::array<ptrdiff_t, 2> p = { { i, j } };
      std::pair<int, ptrdiff_t> v = part.index(p);
      renum[idx] = domain[v.first] + v.second;
    }
  }

  // Assemble
  std::vector<ptrdiff_t> ptr;
  std::vector<ptrdiff_t> col;
  std::vector<double> val;
  std::vector<double> rhs;

  ptr.reserve(chunk + 1);
  col.reserve(chunk * 5);
  val.reserve(chunk * 5);
  rhs.reserve(chunk);

  ptr.push_back(0);

  const double hinv = (n - 1);
  const double h2i = (n - 1) * (n - 1);
  for (ptrdiff_t j = 0, idx = 0; j < n; ++j) {
    for (ptrdiff_t i = 0; i < n; ++i, ++idx) {
      if (renum[idx] < chunk_start || renum[idx] >= chunk_end)
        continue;

      if (j > 0) {
        col.push_back(renum[idx - n]);
        val.push_back(-h2i);
      }

      if (i > 0) {
        col.push_back(renum[idx - 1]);
        val.push_back(-h2i - hinv);
      }

      col.push_back(renum[idx]);
      val.push_back(4 * h2i + hinv);

      if (i + 1 < n) {
        col.push_back(renum[idx + 1]);
        val.push_back(-h2i);
      }

      if (j + 1 < n) {
        col.push_back(renum[idx + n]);
        val.push_back(-h2i);
      }

      rhs.push_back(1);
      ptr.push_back(col.size());
    }
  }

  // Setup
  AlinaParameters* prm = AlinaLib::params_create();

  AlinaLib::params_set_string(prm, "local.coarsening.type", "smoothed_aggregation");
  AlinaLib::params_set_string(prm, "local.relax.type", "spai0");
  AlinaLib::params_set_string(prm, "isolver.type", "bicgstabl");
  AlinaLib::params_set_string(prm, "dsolver.type", "skyline_lu");

  AlinaDistributedSolver* solver = AlinaLib::solver_mpi_create(MPI_COMM_WORLD,
                                                               chunk, ptr.data(), col.data(), val.data(),
                                                               1, constant_deflation, NULL, prm);

  // Solve
  std::vector<double> x(chunk, 0);
  AlinaConvergenceInfo cnv = AlinaLib::solver_mpi_solve(solver, rhs.data(), x.data());

  std::cout << "Iterations: " << cnv.iterations << std::endl
            << "Error:      " << cnv.residual << std::endl;

  // Clean up
  AlinaLib::solver_mpi_destroy(solver);
  AlinaLib::params_destroy(prm);

  if (n <= 4096) {
    if (rank == 0) {
      std::vector<double> X(n2);
      std::copy(x.begin(), x.end(), X.begin());

      for (int i = 1; i < size; ++i)
        MPI_Recv(&X[domain[i]], domain[i + 1] - domain[i], MPI_DOUBLE, i, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::ofstream f("out.dat", std::ios::binary);
      int m = n2;
      f.write((char*)&m, sizeof(int));
      for (ptrdiff_t i = 0; i < n2; ++i)
        f.write((char*)&X[renum[i]], sizeof(double));
    }
    else {
      MPI_Send(x.data(), chunk, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
    }
  }
  return 0;
}

int main(int argc, char* argv[])
{
  return Arcane::Alina::SampleMainContext::execMain(main2, argc, argv);
}
