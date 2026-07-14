// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NumMatrix.h"
#include "arccore/base/MathNumeric.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

//! Make sure NumMatrix is a POD for arrays (no initialization)
static_assert(std::is_same_v<TrueType,ArrayTraits<NumMatrix<double,4,2>>::IsPODType>);

TEST(TestNumMatrix, Real2x2)
{
  RealN2 zero;
  {
    RealN2x2 v1;
    ASSERT_EQ(v1.vx(), zero);
    ASSERT_EQ(v1.vy(), zero);
  }
  {
    double value = 0.2;
    RealN2 r2_value(value);
    RealN2x2 v1(value);
    ASSERT_EQ(v1.vx(), r2_value);
    ASSERT_EQ(v1.vy(), r2_value);
    RealN2x2 v2(v1);
    ASSERT_EQ(v2.vx(), v1.vx());
    ASSERT_EQ(v2.vy(), v1.vy());
    RealN2 rx(3.5, 1.2);
    RealN2 ry(1.6, 2.1);
    v2.setRow(0,rx);
    v2.setRow(1,ry);
    v1 = v2;
    ASSERT_EQ(v2.vx(), rx);
    ASSERT_EQ(v2.vy(), ry);
    ASSERT_EQ(v1, v2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2ElementAccess)
{
  RealN2x2 m;
  m(0, 0) = 1.0;
  m(0, 1) = 2.0;
  m(1, 0) = 3.0;
  m(1, 1) = 4.0;
  ASSERT_EQ(m(0, 0), 1.0);
  ASSERT_EQ(m(0, 1), 2.0);
  ASSERT_EQ(m(1, 0), 3.0);
  ASSERT_EQ(m(1, 1), 4.0);

  // Row access
  RealN2 r0 = m.row(0);
  ASSERT_EQ(r0.vx(), 1.0);
  ASSERT_EQ(r0.vy(), 2.0);
  RealN2 r1 = m.row(1);
  ASSERT_EQ(r1.vx(), 3.0);
  ASSERT_EQ(r1.vy(), 4.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2SetRow)
{
  RealN2x2 m;
  m.setRow(0, RealN2(5.0, 6.0));
  m.setRow(1, RealN2(7.0, 8.0));
  ASSERT_EQ(m(0, 0), 5.0);
  ASSERT_EQ(m(0, 1), 6.0);
  ASSERT_EQ(m(1, 0), 7.0);
  ASSERT_EQ(m(1, 1), 8.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2Fill)
{
  RealN2x2 m;
  m.fill(3.5);
  ASSERT_EQ(m(0, 0), 3.5);
  ASSERT_EQ(m(0, 1), 3.5);
  ASSERT_EQ(m(1, 0), 3.5);
  ASSERT_EQ(m(1, 1), 3.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2Zero)
{
  RealN2x2 m = RealN2x2::zero();
  RealN2 zero;
  ASSERT_EQ(m.vx(), zero);
  ASSERT_EQ(m.vy(), zero);
  ASSERT_TRUE(math::isNearlyZero(m));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2Arithmetic)
{
  RealN2x2 a(RealN2(1.0, 2.0), RealN2(3.0, 4.0));
  RealN2x2 b(RealN2(5.0, 6.0), RealN2(7.0, 8.0));

  // Addition
  RealN2x2 c = a + b;
  ASSERT_EQ(c(0, 0), 6.0);
  ASSERT_EQ(c(0, 1), 8.0);
  ASSERT_EQ(c(1, 0), 10.0);
  ASSERT_EQ(c(1, 1), 12.0);

  // Subtraction
  RealN2x2 d = c - b;
  ASSERT_EQ(d, a);

  // Negation
  RealN2x2 e = -a;
  ASSERT_EQ(e(0, 0), -1.0);
  ASSERT_EQ(e(0, 1), -2.0);

  // Scalar multiplication
  RealN2x2 f = a * 2.0;
  ASSERT_EQ(f(0, 0), 2.0);
  ASSERT_EQ(f(0, 1), 4.0);
  ASSERT_EQ(f(1, 0), 6.0);
  ASSERT_EQ(f(1, 1), 8.0);

  RealN2x2 g = 3.0 * a;
  ASSERT_EQ(g(0, 0), 3.0);
  ASSERT_EQ(g(0, 1), 6.0);

  // Scalar division
  RealN2x2 h = f / 2.0;
  ASSERT_EQ(h, a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2CompoundAssignment)
{
  RealN2x2 a(RealN2(1.0, 2.0), RealN2(3.0, 4.0));
  RealN2x2 b(RealN2(5.0, 6.0), RealN2(7.0, 8.0));

  RealN2x2 c = a;
  c += b;
  ASSERT_EQ(c(0, 0), 6.0);
  ASSERT_EQ(c(0, 1), 8.0);

  c -= b;
  ASSERT_EQ(c, a);

  c *= 2.0;
  ASSERT_EQ(c(0, 0), 2.0);
  ASSERT_EQ(c(0, 1), 4.0);

  RealN2x2 d = a;
  d /= 2.0;
  ASSERT_EQ(d(0, 0), 0.5);
  ASSERT_EQ(d(0, 1), 1.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2ScalarAssignment)
{
  RealN2x2 m;
  m = 5.0;
  ASSERT_EQ(m(0, 0), 5.0);
  ASSERT_EQ(m(0, 1), 5.0);
  ASSERT_EQ(m(1, 0), 5.0);
  ASSERT_EQ(m(1, 1), 5.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real2x2Conversion)
{
  RealN2x2 m(RealN2(1.0, 2.0), RealN2(3.0, 4.0));
  Real2x2 r = static_cast<Real2x2>(m);
  ASSERT_EQ(r.x.x, 1.0);
  ASSERT_EQ(r.x.y, 2.0);
  ASSERT_EQ(r.y.x, 3.0);
  ASSERT_EQ(r.y.y, 4.0);

  // Construction from Real2x2
  RealN2x2 m2(Real2x2(Real2(1.0, 2.0), Real2(3.0, 4.0)));
  ASSERT_EQ(m2(0, 0), 1.0);
  ASSERT_EQ(m2(1, 1), 4.0);

  // Assignment from Real2x2
  RealN2x2 m3;
  m3 = Real2x2(Real2(5.0, 6.0), Real2(7.0, 8.0));
  ASSERT_EQ(m3(0, 0), 5.0);
  ASSERT_EQ(m3(1, 1), 8.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3)
{
  RealN3 zero;
  {
    RealN3x3 v1;
    ASSERT_EQ(v1.vx(), zero);
    ASSERT_EQ(v1.vy(), zero);
    ASSERT_EQ(v1.vz(), zero);
  }
  {
    double value = 0.2;
    RealN3 r3_value(value);
    RealN3x3 v1(value);
    ASSERT_EQ(v1.vx(), r3_value);
    ASSERT_EQ(v1.vy(), r3_value);
    ASSERT_EQ(v1.vz(), r3_value);
    RealN3x3 v2(v1);
    ASSERT_EQ(v2.vx(), v1.vx());
    ASSERT_EQ(v2.vy(), v1.vy());
    ASSERT_EQ(v2.vz(), v1.vz());
    RealN3 rx(3.5, 1.2, -1.5);
    RealN3 ry(1.6, 2.1, -2.3);
    RealN3 rz(-2.3, 1.8, 9.4);
    v2.setRow(0,rx);
    v2.setRow(1,ry);
    v2.setRow(2,rz);
    v1 = v2;
    ASSERT_EQ(v1.vx(), rx);
    ASSERT_EQ(v1.vy(), ry);
    ASSERT_EQ(v1.vz(), rz);
    ASSERT_EQ(v1, v2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3FromColumns)
{
  RealN3x3 m = RealN3x3::fromColumns(1.0, 2.0, 3.0,
                                      4.0, 5.0, 6.0,
                                      7.0, 8.0, 9.0);
  // Columns: (ax,bx,cx) = (1,4,7), (ay,by,cy) = (2,5,8), (az,bz,cz) = (3,6,9)
  // So row 0 = (ax, ay, az) = (1,2,3)
  ASSERT_EQ(m(0, 0), 1.0);
  ASSERT_EQ(m(0, 1), 4.0);
  ASSERT_EQ(m(0, 2), 7.0);
  ASSERT_EQ(m(1, 0), 2.0);
  ASSERT_EQ(m(1, 1), 5.0);
  ASSERT_EQ(m(1, 2), 8.0);
  ASSERT_EQ(m(2, 0), 3.0);
  ASSERT_EQ(m(2, 1), 6.0);
  ASSERT_EQ(m(2, 2), 9.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3FromLines)
{
  RealN3x3 m = RealN3x3::fromLines(1.0, 2.0, 3.0,
                                    4.0, 5.0, 6.0,
                                    7.0, 8.0, 9.0);
  ASSERT_EQ(m(0, 0), 1.0);
  ASSERT_EQ(m(0, 1), 2.0);
  ASSERT_EQ(m(0, 2), 3.0);
  ASSERT_EQ(m(1, 0), 4.0);
  ASSERT_EQ(m(1, 1), 5.0);
  ASSERT_EQ(m(1, 2), 6.0);
  ASSERT_EQ(m(2, 0), 7.0);
  ASSERT_EQ(m(2, 1), 8.0);
  ASSERT_EQ(m(2, 2), 9.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3Conversion)
{
  RealN3x3 m(RealN3(1.0, 2.0, 3.0),
             RealN3(4.0, 5.0, 6.0),
             RealN3(7.0, 8.0, 9.0));
  Real3x3 r = static_cast<Real3x3>(m);
  ASSERT_EQ(r.x.x, 1.0);
  ASSERT_EQ(r.x.y, 2.0);
  ASSERT_EQ(r.x.z, 3.0);
  ASSERT_EQ(r.y.x, 4.0);
  ASSERT_EQ(r.y.y, 5.0);
  ASSERT_EQ(r.y.z, 6.0);
  ASSERT_EQ(r.z.x, 7.0);
  ASSERT_EQ(r.z.y, 8.0);
  ASSERT_EQ(r.z.z, 9.0);

  // Construction from Real3x3
  Real3x3 r3(Real3(1.0, 2.0, 3.0), Real3(4.0, 5.0, 6.0), Real3(7.0, 8.0, 9.0));
  RealN3x3 m2(r3);
  ASSERT_EQ(m2(0, 0), 1.0);
  ASSERT_EQ(m2(1, 1), 5.0);
  ASSERT_EQ(m2(2, 2), 9.0);

  // Assignment from Real3x3
  RealN3x3 m3;
  m3 = r3;
  ASSERT_EQ(m3, m2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3ElementAccess)
{
  RealN3x3 m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      m(i, j) = double(i * 3 + j + 1);

  ASSERT_EQ(m(0, 0), 1.0);
  ASSERT_EQ(m(0, 1), 2.0);
  ASSERT_EQ(m(0, 2), 3.0);
  ASSERT_EQ(m(1, 0), 4.0);
  ASSERT_EQ(m(1, 1), 5.0);
  ASSERT_EQ(m(1, 2), 6.0);
  ASSERT_EQ(m(2, 0), 7.0);
  ASSERT_EQ(m(2, 1), 8.0);
  ASSERT_EQ(m(2, 2), 9.0);

  ASSERT_EQ(m.row(0), RealN3(1.0, 2.0, 3.0));
  ASSERT_EQ(m.row(1), RealN3(4.0, 5.0, 6.0));
  ASSERT_EQ(m.row(2), RealN3(7.0, 8.0, 9.0));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3Arithmetic)
{
  RealN3x3 a(RealN3(1.0, 2.0, 3.0),
             RealN3(4.0, 5.0, 6.0),
             RealN3(7.0, 8.0, 9.0));
  RealN3x3 b(RealN3(9.0, 8.0, 7.0),
             RealN3(6.0, 5.0, 4.0),
             RealN3(3.0, 2.0, 1.0));

  RealN3x3 c = a + b;
  ASSERT_EQ(c(0, 0), 10.0);
  ASSERT_EQ(c(1, 1), 10.0);
  ASSERT_EQ(c(2, 2), 10.0);

  RealN3x3 d = c - b;
  ASSERT_EQ(d, a);

  RealN3x3 e = -a;
  ASSERT_EQ(e(0, 0), -1.0);
  ASSERT_EQ(e(1, 1), -5.0);
  ASSERT_EQ(e(2, 2), -9.0);

  RealN3x3 f = a * 2.0;
  ASSERT_EQ(f(0, 0), 2.0);
  ASSERT_EQ(f(1, 1), 10.0);

  RealN3x3 g = 0.5 * f;
  ASSERT_EQ(g, a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3Fill)
{
  RealN3x3 m;
  m.fill(-1.0);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ASSERT_EQ(m(i, j), -1.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Real3x3NearlyZero)
{
  RealN3x3 m;
  ASSERT_TRUE(math::isNearlyZero(m));
  m(0, 0) = 1e-20;
  ASSERT_TRUE(math::isNearlyZero(m));
  m(0, 0) = 1.0;
  ASSERT_FALSE(math::isNearlyZero(m));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, NonSquare2x5)
{
  using M2x5 = NumMatrix<Real, 2, 5>;

  M2x5 m;
  for (int j = 0; j < 5; ++j) {
    m(0, j) = double(j);
    m(1, j) = double(j * 10);
  }
  for (int j = 0; j < 5; ++j) {
    ASSERT_EQ(m(0, j), double(j));
    ASSERT_EQ(m(1, j), double(j * 10));
  }

  ASSERT_EQ(m.row(0).vx(), 0.0);
  ASSERT_EQ(m.row(0)(4), 4.0);
  ASSERT_EQ(m.row(1)(3), 30.0);

  M2x5 n;
  n.fill(7.0);
  for (int j = 0; j < 5; ++j) {
    ASSERT_EQ(n(0, j), 7.0);
    ASSERT_EQ(n(1, j), 7.0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, NonSquare5x2)
{
  using M5x2 = NumMatrix<Real, 5, 2>;

  M5x2 m;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      m(i, j) = double(i + j);

  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      ASSERT_EQ(m(i, j), double(i + j));

  ASSERT_EQ(m.vx()(1), 1.0);
  ASSERT_EQ(m.vy()(1), 1.0 + 1.0);
  ASSERT_EQ(m.row(3)(1), 4.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumMatrix, Int32x2)
{
  using Int32x2 = NumMatrix<Int32, 2, 2>;

  Int32x2 m;
  ASSERT_EQ(m(0, 0), 0);
  ASSERT_EQ(m(1, 1), 0);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  ASSERT_EQ(m(0, 0), 1);
  ASSERT_EQ(m(1, 1), 4);

  Int32x2 n(Int32x2::VectorType(5, 6), Int32x2::VectorType(7, 8));
  ASSERT_EQ(n(0, 0), 5);
  ASSERT_EQ(n(1, 1), 8);

  Int32x2 p = m + n;
  ASSERT_EQ(p(0, 0), 6);
  ASSERT_EQ(p(0, 1), 8);
  ASSERT_EQ(p(1, 0), 10);
  ASSERT_EQ(p(1, 1), 12);

  p *= 2;
  ASSERT_EQ(p(0, 0), 12);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class NumMatrix<Real, 2, 2>;
template class NumMatrix<Real, 3, 3>;
template class NumMatrix<Real, 1, 1>;
template class NumMatrix<Real, 2, 5>;
template class NumMatrix<Real,5, 2>;
template class NumMatrix<float, 2, 6>;
template class NumMatrix<Int32, 2, 2>;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
