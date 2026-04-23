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

#include <gtest/gtest.h>

#include "arccore/base/PlatformUtils.h"

#include "arccore/alina/Adapters.h"
#include "arccore/alina/BuiltinBackend.h"

#include "arccore/alina/CSRMatrixView.h"
#include "arccore/alina/CSRMatrix.h"

#include "SampleProblemCommon.h"

using namespace Arcane;
using namespace Arcane::Alina;

namespace
{
template <class Val, class Col, class Ptr>
CSRMatrixView<Val, Col, Ptr>
matrixView(CSRMatrix<Val, Col, Ptr>& matrix)
{
  return CSRMatrixView<Val,Col,Ptr>(matrix.nbRow(), matrix.nbNonZero(),
                       matrix.ptr.data(), matrix.col.data(), matrix.val.data());
}

/// Scale matrix values.
template <class Val, class Col, class Ptr, class T> void
doTestScale1(CSRMatrix<Val, Col, Ptr>& A, T s, Int32 nb_loop)
{
  const ptrdiff_t nb_row = backend::nbRow(A);

  std::cout << "DO_TEST_SCALE1 nb_row=" << nb_row
            << " nb_non_zero=" << A.nbNonZero() << "\n";
  Ptr* row_ptr = &A.ptr[0];
  Real t1 = Platform::getRealTime();
  for (Int32 z = 0; z < nb_loop; ++z) {
    for (Int32 i = 0; i < nb_row; ++i) {
      for (ptrdiff_t j = row_ptr[i], e = row_ptr[i + 1]; j < e; ++j)
        A.val[j] *= s;
    }
  }
  Real t2 = Platform::getRealTime();
  std::cout << "Scale1_Time=" << (t2 - t1) << "\n";
}
/// Scale matrix values.
template <class Val, class Col, class Ptr, class T> void
doTestScale2(CSRMatrixView<Val, Col, Ptr> A, T s, Int32 nb_loop)
{
  const ptrdiff_t nb_row = A.nbRow();

  std::cout << "DO_TEST_SCALE2 nb_row=" << nb_row
            << " nb_non_zero=" << A.nbNonZero() << "\n";
  const bool is_verbose = false;
  if (is_verbose) {
    for (Int32 i = 0; i < 5; ++i) {
      for (auto ci : A.rowRange(i))
        std::cout << "X=" << i << " j=" << A.column(ci) << " v=" << A.value(ci) << "\n";
    }
  }
  Real t1 = Platform::getRealTime();
  auto ptrs = A.rowIndexes();
  auto values = A.values(); //.data();
  for (Int32 z = 0; z < nb_loop; ++z) {
    for (Int32 i = 0; i < nb_row; ++i) {
      for (Int32 j = ptrs[i], e = ptrs[i + 1]; j < e; ++j)
        values[j] *= s;
    }
  }
  Real t2 = Platform::getRealTime();
  std::cout << "Scale2_Time=" << (t2 - t1) << "\n";

  for (Int32 z = 0; z < nb_loop; ++z) {
    for (auto row : A.rows()) {
      for (auto ci : row)
        A.value(ci) *= s;
    }
  }
  Real t3 = Platform::getRealTime();
  std::cout << "Scale3_Time=" << (t3 - t2) << "\n";

  {
    Int32 nb_done = 0;
    for (auto row : A.rows()) {
      for ([[maybe_unused]] auto ci : row) {
        ++nb_done;
      }
    }
    ASSERT_EQ(nb_done, A.nbNonZero());
  }

  if (is_verbose) {
    for (Int32 i = 0; i < 5; ++i) {
      for (auto ci : A.rowRange(i))
        std::cout << "I=" << i << " v=" << A.value(ci) << "\n";
    }
  }
}
} // namespace

TEST(alina_test_csr_matrix_view, basic)
{
  std::vector<ptrdiff_t> ptr;
  std::vector<ptrdiff_t> col;
  std::vector<double> val;
  std::vector<double> rhs;

  Int32 nb_square = 64;
  size_t n = sample_problem(nb_square, val, col, ptr, rhs);
  std::cout << "TEST_VIEW_BASIC\n";
  auto A = Alina::adapter::zero_copy(n, ptr.data(), col.data(), val.data());
  auto matrix_view = matrixView(*A);

  ASSERT_EQ(A->nbRow(),matrix_view.nbRow());
  ASSERT_EQ(A->nbNonZero(),matrix_view.nbNonZero());

  Int32 nb_loop = 50;
  if (arccoreIsDebug())
    nb_loop = 5;
  doTestScale1(*A, 2.0, nb_loop);
  doTestScale2(matrixView(*A), 2.0, nb_loop);
}
