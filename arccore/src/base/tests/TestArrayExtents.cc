// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/ArrayExtentsValue.h"
#include "arccore/base/ArrayExtents.h"

#include <iostream>

using namespace Arcane;

TEST(ArrayExtents, Misc)
{
  Int32 n1 = 150;
  Int32 n2 = 120;
  Int32 n3 = 50;
  Int32 n4 = 200;

  {
    ArrayExtents<ExtentsV<Int32, DynExtent>> a(n1);
    Int32 total_size = n1;
    ASSERT_EQ(a.totalNbElement(), total_size);
    ASSERT_EQ(a.extent0(), n1);
  }
  {
    ArrayExtents<ExtentsV<Int32, DynExtent, DynExtent>> a(n1, n2);
    Int32 total_size = n1 * n2;
    ASSERT_EQ(a.totalNbElement(), total_size);
    ASSERT_EQ(a.extent0(), n1);
    ASSERT_EQ(a.extent1(), n2);
  }
  {
    ArrayExtents<ExtentsV<Int32, DynExtent, DynExtent, DynExtent>> a(n1, n2, n3);
    Int32 total_size = n1 * n2 * n3;
    ASSERT_EQ(a.totalNbElement(), total_size);
    ASSERT_EQ(a.extent0(), n1);
    ASSERT_EQ(a.extent1(), n2);
    ASSERT_EQ(a.extent2(), n3);
  }
  {
    ArrayExtents<ExtentsV<Int32, DynExtent, DynExtent, DynExtent, DynExtent>> a(n1, n2, n3, n4);
    Int32 total_size = n1 * n2 * n3 * n4;
    ASSERT_EQ(a.totalNbElement(), total_size);
    ASSERT_EQ(a.extent0(), n1);
    ASSERT_EQ(a.extent1(), n2);
    ASSERT_EQ(a.extent2(), n3);
    ASSERT_EQ(a.extent3(), n4);
  }
}
