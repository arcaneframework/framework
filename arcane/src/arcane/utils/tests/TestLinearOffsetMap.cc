// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/LinearOffsetMap.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(TestLinearOffsetMap, Misc)
{
  {
    LinearOffsetMap<Int32> v1;
    ASSERT_TRUE(v1.getAndRemoveOffset(1) < 0);
    ASSERT_EQ(v1.size(), 0);
  }
  {
    LinearOffsetMap<Int64> v2;
    v2.add(25, 4);
    ASSERT_TRUE(v2.getAndRemoveOffset(32) < 0);
    ASSERT_EQ(v2.getAndRemoveOffset(20), 4);
    ASSERT_EQ(v2.getAndRemoveOffset(5), 24);
    ASSERT_EQ(v2.size(), 0);
  }

  {
    LinearOffsetMap<Int64> v3;
    v3.add(25, 4);
    v3.add(25, 30);
    ASSERT_EQ(v3.getAndRemoveOffset(18), 4);
    ASSERT_EQ(v3.getAndRemoveOffset(25), 30);
    ASSERT_EQ(v3.size(), 1);
    ASSERT_EQ(v3.getAndRemoveOffset(4), 22);
    ASSERT_EQ(v3.size(), 1);
    ASSERT_EQ(v3.getAndRemoveOffset(3), 26);
    ASSERT_EQ(v3.size(), 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
