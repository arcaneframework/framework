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

#include <boost/scope_exit.hpp>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/IO.h"
#include "arccore/alina/Profiler.h"

using namespace Arcane;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  BOOST_SCOPE_EXIT(void)
  {
    MPI_Finalize();
  }
  BOOST_SCOPE_EXIT_END

  Alina::mpi_communicator comm(MPI_COMM_WORLD);

  int n = 16;
  int chunk_len = (n + comm.size - 1) / comm.size;
  int chunk_beg = std::min(n, chunk_len * comm.rank);
  int chunk_end = std::min(n, chunk_len * (comm.rank + 1));
  int chunk = chunk_end - chunk_beg;

  std::vector<int> ptr;
  ptr.reserve(chunk + 1);
  ptr.push_back(0);
  std::vector<int> col;
  col.reserve(chunk * 4);
  std::vector<double> val;
  val.reserve(chunk * 4);

  for (int i = 0, j = chunk_beg; i < chunk; ++i, ++j) {
    if (j > 0) {
      col.push_back(j - 1);
      val.push_back(-1);
    }

    col.push_back(j);
    val.push_back(2);

    if (j + 1 < n) {
      col.push_back(j + 1);
      val.push_back(-1);
    }

    if (j + 5 < n) {
      col.push_back(j + 5);
      val.push_back(-0.1);
    }

    ptr.push_back(col.size());
  }

  typedef Alina::BuiltinBackend<double> Backend;
  typedef Alina::DistributedMatrix<Backend> Matrix;

  Matrix A(comm, std::tie(chunk, ptr, col, val), chunk);

  {
    std::ostringstream fname;
    fname << "A_loc_" << comm.rank << ".mtx";
    Alina::IO::mm_write(fname.str(), *A.local());
  }

  {
    std::ostringstream fname;
    fname << "A_rem_" << comm.rank << ".mtx";
    Alina::IO::mm_write(fname.str(), *A.remote());
  }

  auto B = transpose(A);

  {
    std::ostringstream fname;
    fname << "B_loc_" << comm.rank << ".mtx";
    Alina::IO::mm_write(fname.str(), *B->local());
  }

  {
    std::ostringstream fname;
    fname << "B_rem_" << comm.rank << ".mtx";
    Alina::IO::mm_write(fname.str(), *B->remote());
  }
}
