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

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/StaticMatrix.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/IO.h"

#include "arccore/alina/Profiler.h"

#include <iostream>
#include <vector>

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
  // In case of block, Rhs may be a StaticMatrix.
  using Rhs = math::rhs_of<Val>::type;
  using ScalarRhs = math::scalar_of<Rhs>::type;
  int nb_scalar_for_rhs = sizeof(Rhs) / sizeof(ScalarRhs);

  Alina::mpi_communicator comm(MPI_COMM_WORLD);
  IMessagePassingMng* pm = comm.m_message_passing_mng.get();

  int n = 16;
  int chunk_len = (n + comm.size - 1) / comm.size;
  int chunk_beg = std::min(n, chunk_len * comm.rank);
  int chunk_end = std::min(n, chunk_len * (comm.rank + 1));
  int chunk = chunk_end - chunk_beg;

  UniqueArray<int> chunks(comm.size);
  ConstArrayView<int> sent_chunk(1,&chunk);
  mpAllGather(pm,sent_chunk,chunks);

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

  // Because Rhs may be a StaticMatrix and it is not a basic type
  // we convert it to an array of basic type (which is math::scalar_of<Rhs>::type).
  UniqueArray<ScalarRhs> scalarX;
  UniqueArray<ScalarRhs> scalarR;
  Span<const ScalarRhs> scalar_x_view(reinterpret_cast<ScalarRhs*>(x.data()), x.size() * nb_scalar_for_rhs);
  Span<const ScalarRhs> scalar_r_view(reinterpret_cast<ScalarRhs*>(y.data()), y.size() * nb_scalar_for_rhs);
  mpGatherVariable(pm, scalar_x_view, scalarX, 0);
  mpGatherVariable(pm, scalar_r_view, scalarR, 0);

  if (comm.rank == 0) {
    std::cout << "Doing verification size=" << scalarX.size() << "\n";
    Rhs* scalar_x_begin = reinterpret_cast<Rhs*>(scalarX.data());
    std::vector<Rhs> X(scalar_x_begin, scalar_x_begin + n);

    Rhs* scalar_r_begin = reinterpret_cast<Rhs*>(scalarR.data());
    std::vector<Rhs> R(scalar_r_begin, scalar_r_begin + n);

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

  test<double>();
  test<Alina::StaticMatrix<double, 2, 2>>();
  MPI_Finalize();
}
