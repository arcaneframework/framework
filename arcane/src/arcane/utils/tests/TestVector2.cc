// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/Vector2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(TestVector2, Misc)
{
  {
    Int64x2 v0;
    ASSERT_EQ(v0.x, 0);
    ASSERT_EQ(v0.y, 0);
  }
  {
    Int64x2 v0;
    Int64x2 v1(2, 7);
    ASSERT_EQ(v1.x, 2);
    ASSERT_EQ(v1.y, 7);
    Int64x2 v2(v1);
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
    Int64x2 v3(1, 5);
    Int64x2 v4(2, 3);
    Int64x2 v5(2, 7);
    ASSERT_TRUE(v3 < v2);
    ASSERT_TRUE(v4 < v2);
  }
  {
    Int64x2 v2({1});
    ASSERT_EQ(v2.x, 1);
    ASSERT_EQ(v2.y, 0);
    Int64x2 v3({1,2});
    ASSERT_EQ(v3.x, 1);
    ASSERT_EQ(v3.y, 2);
    Int64x2 v4({1,2,3});
    ASSERT_EQ(v4.x, 1);
    ASSERT_EQ(v4.y, 2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
