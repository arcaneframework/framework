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

#include "arccore/alina/MessagePassingUtils.h"
#include "arccore/alina/AlinaLib.h"

#include "./SampleProblemCommon.h"

#include <gtest/gtest.h>

double constant_deflation(int, ptrdiff_t, void*)
{
  return 1;
}

using namespace Arcane;

TEST(alina_test_mpi, DistributedAlinaLib)
{
  Alina::mpi_communicator world(MPI_COMM_WORLD);

  int comm_rank = world.rank;
  int comm_size = world.size;

  const Int32 n = 64;

  // For 32 bit indexing
  using ColumnType = Int32;
  // For 64 bit indexing
  // using ColumnType = Int64;

  std::vector<ColumnType> ptr;
  std::vector<ColumnType> col;
  std::vector<double> val;
  std::vector<double> rhs;

  Int32 chunk = sample_problem_distributed(comm_rank, comm_size, n, 1, ptr, col, val, rhs);

  // Setup
  AlinaParameters prm;

  prm.setString("local.coarsening.type", "smoothed_aggregation");
  prm.setString("local.relax.type", "spai0");
  prm.setString("isolver.type", "bicgstabl");
  prm.setString("dsolver.type", "skyline_lu");

  UniqueArray<double> x(rhs.size(), 0.0);

  // Solve
  {
    AlinaCSRMatrixView matrix_view(chunk, ptr.data(), col.data(), val.data());
    AlinaDistributedSolver solver(MPI_COMM_WORLD, matrix_view,
                                  1, constant_deflation, nullptr, prm);
    SmallSpan<const double> rhs_view(rhs.data(), rhs.size());
    AlinaConvergenceInfo cnv = solver.solve(rhs_view, x.smallSpan());

    std::cout << "Iterations: " << cnv.iterations << std::endl
              << "Error:      " << cnv.residual << std::endl;
  }
}
