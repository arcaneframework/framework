// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/PlatformUtils.h"
#include "arccore/base/ArrayExtentsValue.h"
#include "arccore/base/ArrayExtents.h"

#include "arccore/base/ForLoopRanges.h"

#include <iostream>

using namespace Arcane;

struct XYZ
{
  Int32 x;
  Int32 y;
  Int32 z;
  Int32 yz = y * z;

  XYZ(Int32 x_, Int32 y_, Int32 z_)
  : x(x_)
  , y(y_)
  , z(z_)
  {
    yz = y * z;
  }
  std::array<Int32, 3> getIndices1(Int32 index)
  {
    Int32 i2 = impl::fastmod(index, z);
    Int32 fac = z;
    Int32 i1 = impl::fastmod(index / fac, y);
    fac *= y;
    Int32 i0 = index / fac;
    return { i0, i1, i2 };
  }

  std::array<Int32, 3> getIndices2(Int32 index)
  {
    Int32 i = index / (y * z);
    index %= (y * z);
    Int32 j = index / z;
    Int32 k = index % z;
    return { i, j, k };
  }
  std::array<Int32, 3> getIndices3(Int32 index)
  {
    Int32 i = index / static_cast<unsigned int>(y * z);
    index %= y * z;
    Int32 j = index / static_cast<unsigned int>(z);
    Int32 k = index % z;
    return { i, j, k };
  }
  std::array<Int32, 3> getIndices4(Int32 index)
  {
    UInt32 uz = static_cast<unsigned int>(z);
    Int32 i = index / yz;
    index %= yz;
    Int32 j = index / uz;
    Int32 k = index % uz;
    return { i, j, k };
  }
};

TEST(GetIndices, Versions)
{
  Int32 n0 = 150;
  Int32 n1 = 120;
  Int32 n2 = 50;
  Int32 total_size = n0 * n1 * n2;
  ArrayExtents<ExtentsV<Int32, DynExtent, DynExtent, DynExtent>> xv(n0, n1, n2);
  XYZ xv2(n0, n1, n2);
  for (Int32 i = 0; i < total_size; ++i) {
    auto x = xv2.getIndices4(i);
    auto y = xv.getIndices(i);
    ASSERT_EQ(x[0], y[0]);
    ASSERT_EQ(x[1], y[1]);
    ASSERT_EQ(x[2], y[2]);
  }
  Int32 nb_loop = 500;
  if (arccoreIsDebug())
    nb_loop = 10;
  double x = Platform::getRealTime();
  Int64 total0 = 0;
  for (Int32 k = 0; k < nb_loop; ++k) {
    for (Int32 i = 0; i < total_size; ++i) {
      auto x = xv.getIndices(i);
      total0 += x[0] + x[1] + x[2];
    }
  }
  double t0 = Platform::getRealTime() - x;
  std::cerr << "TOTAL_0=" << total0 << " time0=" << t0 << "\n";

  x = Platform::getRealTime();
  Int64 total1 = 0;
  for (Int32 k = 0; k < nb_loop; ++k) {
    for (Int32 i = 0; i < total_size; ++i) {
      auto x = xv2.getIndices1(i);
      total1 += x[0] + x[1] + x[2];
    }
  }
  ASSERT_EQ(total0, total1);
  double t1 = Platform::getRealTime() - x;
  std::cerr << "TOTAL_1=" << total1 << " time1=" << t1 << "\n";

  x = Platform::getRealTime();
  Int64 total2 = 0;
  for (Int32 k = 0; k < nb_loop; ++k) {
    for (Int32 i = 0; i < total_size; ++i) {
      auto x = xv2.getIndices2(i);
      total2 += x[0] + x[1] + x[2];
    }
  }
  ASSERT_EQ(total0, total2);
  double t2 = Platform::getRealTime() - x;
  std::cerr << "TOTAL_2=" << total2 << " time2=" << t2 << "\n";

  x = Platform::getRealTime();
  Int64 total3 = 0;
  for (Int32 k = 0; k < nb_loop; ++k) {
    for (Int32 i = 0; i < total_size; ++i) {
      auto x = xv2.getIndices3(i);
      total3 += x[0] + x[1] + x[2];
    }
  }
  ASSERT_EQ(total0, total3);
  double t3 = Platform::getRealTime() - x;
  std::cerr << "TOTAL_3=" << total3 << " time3=" << t3 << "\n";

  x = Platform::getRealTime();
  Int64 total4 = 0;
  for (Int32 k = 0; k < nb_loop; ++k) {
    for (Int32 i = 0; i < total_size; ++i) {
      auto x = xv2.getIndices4(i);
      total4 += x[0] + x[1] + x[2];
    }
  }
  ASSERT_EQ(total0, total4);
  double t4 = Platform::getRealTime() - x;
  std::cerr << "TOTAL_4=" << total4 << " time3=" << t4 << "\n";
}
