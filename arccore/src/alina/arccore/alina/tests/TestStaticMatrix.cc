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

#include "arccore/alina/StaticMatrix.h"

using namespace Arcane;

TEST(alina_test_static_matrix, sum)
{
  Alina::StaticMatrix<int, 2, 2> a = { { 1, 2, 3, 4 } };
  Alina::StaticMatrix<int, 2, 2> b = { { 4, 3, 2, 1 } };
  Alina::StaticMatrix<int, 2, 2> c = a + b;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      ASSERT_EQ(c(i, j), 5);
}

TEST(alina_test_static_matrix, minus)
{
  Alina::StaticMatrix<int, 2, 2> a = { { 5, 5, 5, 5 } };
  Alina::StaticMatrix<int, 2, 2> b = { { 4, 3, 2, 1 } };
  Alina::StaticMatrix<int, 2, 2> c = a - b;

  for (int i = 0; i < 4; ++i)
    ASSERT_EQ(c(i), i + 1);
}

TEST(alina_test_static_matrix, product)
{
  Alina::StaticMatrix<int, 2, 2> a = { { 2, 1, 1, 2 } };
  Alina::StaticMatrix<int, 2, 2> c = a * a;

  ASSERT_EQ(c(0, 0), 5);
  ASSERT_EQ(c(0, 1), 4);
  ASSERT_EQ(c(1, 0), 4);
  ASSERT_EQ(c(1, 1), 5);
}

TEST(alina_test_static_matrix, scale)
{
  Alina::StaticMatrix<int, 2, 2> a = { { 1, 2, 3, 4 } };
  Alina::StaticMatrix<int, 2, 2> c = 2 * a;

  for (int i = 0; i < 4; ++i)
    ASSERT_EQ(c(i), 2 * (i + 1));
}

TEST(alina_test_static_matrix, inner_product)
{
  Alina::StaticMatrix<int, 2, 1> a = { { 1, 2 } };
  int c = Alina::math::inner_product(a, a);

  ASSERT_EQ(c, 5);
}

TEST(alina_test_static_matrix, inverse)
{
  Alina::StaticMatrix<double, 2, 2> a = { { 2.0, -1.0, -1.0, 2.0 } };
  Alina::StaticMatrix<double, 2, 2> b = Alina::math::inverse(a);
  Alina::StaticMatrix<double, 2, 2> c = b * a;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      ASSERT_NEAR(c(i, j) - (i == j), 0.0, 1e-8);
}

TEST(alina_test_static_matrix, inverse_pivoting)
{
  Alina::StaticMatrix<double, 4, 4> a{ {
  1,
  -0.1,
  -0.028644256,
  0.25684664,
  1,
  -0.1,
  -0.025972342,
  0.25663863,
  1,
  -0.095699158,
  -0.029327056,
  0.25554974,
  1,
  -0.09543351,
  -0.026189496,
  0.25796741,
  } };
  Alina::StaticMatrix<double, 4, 4> b = Alina::math::inverse(a);
  Alina::StaticMatrix<double, 4, 4> c = b * a;

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      ASSERT_NEAR(c(i, j) - (i == j), 0.0, 1e-8);
}

TEST(alina_test_static_matrix, inverse_pivoting_2)
{
  Alina::StaticMatrix<double, 4, 4> a{
    {
    0, 1, 0, 0,
    0, 0, 1, 0,
    1, 0, 0, 0,
    0, 0, 0, 1,
    }
  };
  Alina::StaticMatrix<double, 4, 4> b = Alina::math::inverse(a);
  Alina::StaticMatrix<double, 4, 4> c = b * a;

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      ASSERT_NEAR(c(i, j) - (i == j), 0.0, 1e-8);
}
