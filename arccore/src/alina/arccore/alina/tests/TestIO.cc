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

#include "arccore/alina/IO.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/Profiler.h"
#include "SampleProblemCommon.h"

using namespace Arcane;

namespace
{
Arcane::Alina::Profiler prof;
}

TEST(alina_test_io, io_mm)
{
  std::vector<ptrdiff_t> ptr, ptr2;
  std::vector<ptrdiff_t> col, col2;
  std::vector<double> val, val2;
  std::vector<double> rhs, rhs2;

  size_t n = sample_problem(16, val, col, ptr, rhs);

  auto A = std::tie(n, ptr, col, val);

  Alina::IO::mm_write("test_io_crs.mm", A);
  Alina::IO::mm_write("test_io_vec.mm", rhs.data(), n, 1);

  size_t rows, cols;
  std::tie(rows, cols) = Alina::IO::mm_reader("test_io_crs.mm")(ptr2, col2, val2);

  ASSERT_EQ(n, rows);
  ASSERT_EQ(n, cols);
  ASSERT_EQ(ptr.back(), ptr2.back());
  for (size_t i = 0; i < n; ++i) {
    ASSERT_EQ(ptr[i], ptr2[i]);
    for (ptrdiff_t j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
      ASSERT_EQ(col[j], col2[j]);
      ASSERT_NEAR(val[j] - val2[j], 0.0, 1e-12);
    }
  }

  std::tie(rows, cols) = Alina::IO::mm_reader("test_io_vec.mm")(rhs2);
  ASSERT_EQ(n, rows);
  ASSERT_EQ(1, cols);
  for (size_t i = 0; i < n; ++i) {
    ASSERT_NEAR(rhs[i] - rhs2[i], 0.0, 1e-12);
  }
}
