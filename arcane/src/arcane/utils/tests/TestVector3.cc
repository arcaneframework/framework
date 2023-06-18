// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
