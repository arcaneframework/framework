// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/Vector3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(TestVector3, Misc)
{
  using Int64x3 = Vector3<Int64>;

  {
    Int64x3 v0;
    ASSERT_EQ(v0.x, 0);
    ASSERT_EQ(v0.y, 0);
    ASSERT_EQ(v0.z, 0);
  }
  {
    Int64x3 v0;
    Int64x3 v1(2, 7, -5);
    ASSERT_EQ(v1.x, 2);
    ASSERT_EQ(v1.y, 7);
    ASSERT_EQ(v1.z, -5);
    Int64x3 v2(v1);
    ASSERT_TRUE(v1 == v2);
    ASSERT_FALSE(v1 != v2);
    ASSERT_TRUE(v2 != v0);
    v0 = v2;
    ASSERT_TRUE(v2 == v0);
    std::cout << "V0=" << v0 << "\n";
    std::cout << "V1=" << v1 << "\n";
    std::cout << "V2=" << v2 << "\n";
    ASSERT_FALSE(v0 < v2);
    ASSERT_FALSE(v2 < v0);
    Int64x3 v3(1, 5, 4);
    Int64x3 v4(2, 3, 2);
    Int64x3 v5(2, 7, 2);
    ASSERT_TRUE(v3 < v2);
    ASSERT_TRUE(v4 < v2);
    ASSERT_TRUE(v4 < v5);
  }
  {
    Int64x3 v1({ 1 });
    ASSERT_EQ(v1.x, 1);
    ASSERT_EQ(v1.y, 0);
    ASSERT_EQ(v1.z, 0);
    Int64x3 v2({ 1, 2 });
    ASSERT_EQ(v2.x, 1);
    ASSERT_EQ(v2.y, 2);
    ASSERT_EQ(v2.z, 0);
    Int64x3 v3({ 1, 2, 3 });
    ASSERT_EQ(v3.x, 1);
    ASSERT_EQ(v3.y, 2);
    ASSERT_EQ(v3.z, 3);
    Int64x3 v4({ 1, 2, 3, 4 });
    ASSERT_EQ(v4.x, 1);
    ASSERT_EQ(v4.y, 2);
    ASSERT_EQ(v4.z, 3);
  }
  {
    Int64x3 v1(1, -2, 4);
    Int64x3 v2(3, 5, -2);
    Int64x3 sum_v1_v2(4, 3, 2);
    Int64x3 mul_v1_4(4, -8, 16);
    ASSERT_EQ((v1 + 7), Int64x3(8, 5, 11));
    ASSERT_EQ((7 + v1), Int64x3(8, 5, 11));
    ASSERT_EQ((v1 - 9), Int64x3(-8, -11, -5));
    ASSERT_EQ((v1 + v2), sum_v1_v2);
    Int64x3 v3 = v1;
    v3 += v2;
    ASSERT_EQ(v3, sum_v1_v2);
    v3 -= v2;
    ASSERT_EQ(v3, v1);
    v3 -= -5;
    ASSERT_EQ(v3, (v1 + 5));
    Int64x3 v4 = (v1 * 4);
    ASSERT_EQ(v4, mul_v1_4);
    ASSERT_EQ((4 * v1), mul_v1_4);
    ASSERT_EQ((v4 / 4), v1);
    Int64x3 v5 = v4;
    v5 /= 4;
    ASSERT_EQ(v5, v1);
    Int64x3 v6 = -v1;
    ASSERT_EQ(v6, Int64x3(-1, 2, -4));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
