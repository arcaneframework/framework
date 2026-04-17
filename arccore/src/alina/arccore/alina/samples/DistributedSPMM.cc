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
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/IO.h"

#include "arccore/alina/Profiler.h"

using namespace Arcane;

namespace math = Alina::math;

template <class Val>
void assemble(int n, int beg, int end,
              std::vector<int>& ptr,
              std::vector<int>& col,
              std::vector<Val>& val)
{
  int chunk = end - beg;

  ptr.clear();
  ptr.reserve(chunk + 1);
  ptr.push_back(0);
  col.clear();
  col.reserve(chunk * 4);
  val.clear();
  val.reserve(chunk * 4);

  for (int j = beg, i = 0; j < end; ++j, ++i) {
    if (j > 0) {
      col.push_back(j - 1);
      val.push_back(-math::identity<Val>());
    }

    col.push_back(j);
    val.push_back(2 * math::identity<Val>());

    if (j + 1 < n) {
      col.push_back(j + 1);
      val.push_back(-math::identity<Val>());
    }

    if (j + 5 < n) {
      col.push_back(j + 5);
      val.push_back(-0.1 * math::identity<Val>());
    }

    ptr.push_back(col.size());
  }
}

template <class Val>
void test()
{
  typedef typename math::rhs_of<Val>::type Rhs;

  Alina::mpi_communicator comm(MPI_COMM_WORLD);

  int n = 16;
  int chunk_len = (n + comm.size - 1) / comm.size;
  int chunk_beg = std::min(n, chunk_len * comm.rank);
  int chunk_end = std::min(n, chunk_len * (comm.rank + 1));
  int chunk = chunk_end - chunk_beg;

  std::vector<int> chunks(comm.size);
  MPI_Allgather(&chunk, 1, MPI_INT, &chunks[0], 1, MPI_INT, comm);
  std::vector<int> displ(comm.size, 0);
  for (int i = 1; i < comm.size; ++i)
    displ[i] = displ[i - 1] + chunks[i - 1];

  std::vector<int> ptr;
  std::vector<int> col;
  std::vector<Val> val;
  std::vector<Rhs> x(chunk);
  std::vector<Rhs> y(chunk);

  assemble(n, chunk_beg, chunk_end, ptr, col, val);

  for (int i = 0; i < chunk; ++i)
    x[i] = math::constant<Rhs>(drand48());

  typedef Alina::BuiltinBackend<Val> Backend;
  typedef Alina::DistributedMatrix<Backend> Matrix;

  Matrix A(comm, std::tie(chunk, ptr, col, val), chunk);

  auto B = Alina::product(A, A);
  B->move_to_backend();

  Alina::backend::spmv(1, *B, x, 0, y);

  std::vector<Rhs> X(n), R(n);
  MPI_Gatherv(&x[0], chunk, Alina::mpi_datatype<Rhs>(), &X[0], &chunks[0], &displ[0], Alina::mpi_datatype<Rhs>(), 0, comm);
  MPI_Gatherv(&y[0], chunk, Alina::mpi_datatype<Rhs>(), &R[0], &chunks[0], &displ[0], Alina::mpi_datatype<Rhs>(), 0, comm);

  if (comm.rank == 0) {
    std::vector<Rhs> Y(n);
    assemble(n, 0, n, ptr, col, val);

    Alina::CSRMatrix<Val> A(std::tie(n, ptr, col, val));
    Alina::backend::spmv(1, *product(A, A), X, 0, Y);

    double s = 0;
    for (int i = 0; i < n; ++i) {
      double d = math::norm(R[i] - Y[i]);
      s += d * d;
    }
    std::cout << "Error: " << s << std::endl;
  }
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  BOOST_SCOPE_EXIT(void)
  {
    MPI_Finalize();
  }
  BOOST_SCOPE_EXIT_END

  test<double>();
  test<Alina::StaticMatrix<double, 2, 2>>();
}
