// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NumVector.h"
#include "arcane/utils/NumMatrix.h"
#include "arccore/base/MathNumeric.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

//! Make sure NumVector is a POD for arrays (no initialization)
static_assert(std::is_same_v<TrueType,ArrayTraits<NumVector<double,4>>::IsPODType>);

TEST(TestNumVector, RealN2)
{
  std::cout << "   sizeof(NumVector<double,2>) = " << sizeof(NumVector<double, 2>) << "\n";
  std::cout << "   sizeof(NumVector<double,3>) = " << sizeof(NumVector<double, 3>) << "\n";
  {
    RealN2 v1;
    ASSERT_EQ(v1.vx(), 0.0);
    ASSERT_EQ(v1.vy(), 0.0);
  }
  {
    double value = 0.2;
    RealN2 v1(value);
    ASSERT_EQ(v1.vx(), value);
    ASSERT_EQ(v1.vy(), value);
    RealN2 v2(v1);
    ASSERT_EQ(v2.vx(), value);
    ASSERT_EQ(v2.vy(), value);
    v2.vx() = 3.5;
    v2.vy() = 1.2;
    v1 = v2;
    ASSERT_EQ(v1.vx(), 3.5);
    ASSERT_EQ(v1.vy(), 1.2);
    ASSERT_EQ(v1, v2);
  }
  {
    double value = 0.3;
    RealN2 v2;
    v2 = value;
    ASSERT_EQ(v2.vx(), value);
    ASSERT_EQ(v2.vx(), value);
  }
  // Operator + and -
  {
    RealN2 v2(1.2, 4.5);
    RealN2 v1(1.3, 2.3);
    RealN2 v3 = v1 + v2;
    ASSERT_EQ(v3.vx(), 2.5);
    ASSERT_EQ(v3.vy(), 6.8);
    RealN2 v4 = v3 - v2;
    ASSERT_EQ(v4, v1);
  }
  // Operator * and /
  {
    RealN2 v2(1.2, 4.5);
    RealN2 v4 = v2 * 2.3;
    ASSERT_EQ(v4.vx(), v2.vx() * 2.3);
    ASSERT_EQ(v4.vy(), v2.vy() * 2.3);
    RealN2 v5 = v4 / 2.4;
    ASSERT_EQ(v5.vx(), v4.vx() / 2.4);
    ASSERT_EQ(v5.vy(), v4.vy() / 2.4);
  }
}

TEST(TestNumVector, Real3)
{
  {
    RealN3 v1;
    ASSERT_EQ(v1.vx(), 0.0);
    ASSERT_EQ(v1.vy(), 0.0);
    ASSERT_EQ(v1.vz(), 0.0);
  }
  {
    double value = 0.2;
    RealN3 v1(value);
    ASSERT_EQ(v1.vx(), value);
    ASSERT_EQ(v1.vy(), value);
    ASSERT_EQ(v1.vz(), value);
    RealN3 v2(v1);
    ASSERT_EQ(v2.vx(), value);
    ASSERT_EQ(v2.vy(), value);
    ASSERT_EQ(v2.vz(), value);
    v2.vx() = 3.5;
    v2.vy() = 1.2;
    v2.vz() = -1.5;
    v1 = v2;
    ASSERT_EQ(v1.vx(), 3.5);
    ASSERT_EQ(v1.vy(), 1.2);
    ASSERT_EQ(v1.vz(), -1.5);
    ASSERT_EQ(v1, v2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2ElementAccess)
{
  RealN2 v;
  v(0) = 1.5;
  v(1) = 2.5;
  ASSERT_EQ(v(0), 1.5);
  ASSERT_EQ(v(1), 2.5);
  ASSERT_EQ(v[0], 1.5);
  ASSERT_EQ(v[1], 2.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2Fill)
{
  RealN2 v;
  v.fill(4.2);
  ASSERT_EQ(v.vx(), 4.2);
  ASSERT_EQ(v.vy(), 4.2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2Zero)
{
  RealN2 v = RealN2::zero();
  ASSERT_EQ(v.vx(), 0.0);
  ASSERT_EQ(v.vy(), 0.0);
  ASSERT_TRUE(math::isNearlyZero(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2CompoundAssignment)
{
  RealN2 v1(1.0, 2.0);
  RealN2 v2(3.0, 4.0);

  v1 += v2;
  ASSERT_EQ(v1.vx(), 4.0);
  ASSERT_EQ(v1.vy(), 6.0);

  v1 -= v2;
  ASSERT_EQ(v1.vx(), 1.0);
  ASSERT_EQ(v1.vy(), 2.0);

  v1 *= 2.0;
  ASSERT_EQ(v1.vx(), 2.0);
  ASSERT_EQ(v1.vy(), 4.0);

  v1 /= 2.0;
  ASSERT_EQ(v1.vx(), 1.0);
  ASSERT_EQ(v1.vy(), 2.0);

  v1 += 10.0;
  ASSERT_EQ(v1.vx(), 11.0);
  ASSERT_EQ(v1.vy(), 12.0);

  v1 -= 1.0;
  ASSERT_EQ(v1.vx(), 10.0);
  ASSERT_EQ(v1.vy(), 11.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2UnaryNegation)
{
  RealN2 v1(1.5, -2.5);
  RealN2 v2 = -v1;
  ASSERT_EQ(v2.vx(), -1.5);
  ASSERT_EQ(v2.vy(), 2.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2Norm)
{
  RealN2 v(3.0, 4.0);
  ASSERT_EQ(math::squareNormL2(v), 25.0);
  ASSERT_EQ(math::normL2(v), 5.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2Absolute)
{
  RealN2 v1(-1.5, 2.5);
  RealN2 v2 = v1.absolute();
  ASSERT_EQ(v2.vx(), 1.5);
  ASSERT_EQ(v2.vy(), 2.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2NearlyZero)
{
  RealN2 v;
  ASSERT_TRUE(math::isNearlyZero(v));
  v.vx() = 1e-20;
  ASSERT_TRUE(math::isNearlyZero(v));
  v.vx() = 1.0;
  ASSERT_FALSE(math::isNearlyZero(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2Conversion)
{
  RealN2 v(1.5, 2.5);
  Real2 r = static_cast<Real2>(v);
  ASSERT_EQ(r.x, 1.5);
  ASSERT_EQ(r.y, 2.5);

  RealN2 v2(Real2(3.5, 4.5));
  ASSERT_EQ(v2.vx(), 3.5);
  ASSERT_EQ(v2.vy(), 4.5);

  RealN2 v3;
  v3 = Real2(5.5, 6.5);
  ASSERT_EQ(v3.vx(), 5.5);
  ASSERT_EQ(v3.vy(), 6.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN2NotEqual)
{
  RealN2 v1(1.0, 2.0);
  RealN2 v2(1.0, 3.0);
  ASSERT_NE(v1, v2);
  ASSERT_TRUE(v1 != v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3ElementAccess)
{
  RealN3 v;
  v(0) = 1.0;
  v(1) = 2.0;
  v(2) = 3.0;
  ASSERT_EQ(v(0), 1.0);
  ASSERT_EQ(v(1), 2.0);
  ASSERT_EQ(v(2), 3.0);
  ASSERT_EQ(v[0], 1.0);
  ASSERT_EQ(v[1], 2.0);
  ASSERT_EQ(v[2], 3.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3Fill)
{
  RealN3 v;
  v.fill(-1.0);
  ASSERT_EQ(v.vx(), -1.0);
  ASSERT_EQ(v.vy(), -1.0);
  ASSERT_EQ(v.vz(), -1.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3Zero)
{
  RealN3 v = RealN3::zero();
  ASSERT_EQ(v.vx(), 0.0);
  ASSERT_EQ(v.vy(), 0.0);
  ASSERT_EQ(v.vz(), 0.0);
  ASSERT_TRUE(math::isNearlyZero(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3CompoundAssignment)
{
  RealN3 v1(1.0, 2.0, 3.0);
  RealN3 v2(4.0, 5.0, 6.0);

  v1 += v2;
  ASSERT_EQ(v1.vx(), 5.0);
  ASSERT_EQ(v1.vy(), 7.0);
  ASSERT_EQ(v1.vz(), 9.0);

  v1 -= v2;
  ASSERT_EQ(v1, RealN3(1.0, 2.0, 3.0));

  v1 *= 2.0;
  ASSERT_EQ(v1.vx(), 2.0);
  ASSERT_EQ(v1.vy(), 4.0);
  ASSERT_EQ(v1.vz(), 6.0);

  v1 /= 2.0;
  ASSERT_EQ(v1, RealN3(1.0, 2.0, 3.0));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3UnaryNegation)
{
  RealN3 v1(1.0, -2.0, 3.0);
  RealN3 v2 = -v1;
  ASSERT_EQ(v2.vx(), -1.0);
  ASSERT_EQ(v2.vy(), 2.0);
  ASSERT_EQ(v2.vz(), -3.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3Norm)
{
  RealN3 v(1.0, 2.0, 2.0);
  ASSERT_EQ(math::squareNormL2(v), 9.0);
  ASSERT_EQ(math::normL2(v), 3.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3Absolute)
{
  RealN3 v1(-1.0, 2.0, -3.0);
  RealN3 v2 = v1.absolute();
  ASSERT_EQ(v2.vx(), 1.0);
  ASSERT_EQ(v2.vy(), 2.0);
  ASSERT_EQ(v2.vz(), 3.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3NearlyZero)
{
  RealN3 v;
  ASSERT_TRUE(math::isNearlyZero(v));
  v.vz() = 1e-20;
  ASSERT_TRUE(math::isNearlyZero(v));
  v.vz() = 1.0;
  ASSERT_FALSE(math::isNearlyZero(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3Conversion)
{
  RealN3 v(1.0, 2.0, 3.0);
  Real3 r = static_cast<Real3>(v);
  ASSERT_EQ(r.x, 1.0);
  ASSERT_EQ(r.y, 2.0);
  ASSERT_EQ(r.z, 3.0);

  RealN3 v2(Real3(4.0, 5.0, 6.0));
  ASSERT_EQ(v2.vx(), 4.0);
  ASSERT_EQ(v2.vy(), 5.0);
  ASSERT_EQ(v2.vz(), 6.0);

  RealN3 v3;
  v3 = Real3(7.0, 8.0, 9.0);
  ASSERT_EQ(v3.vx(), 7.0);
  ASSERT_EQ(v3.vy(), 8.0);
  ASSERT_EQ(v3.vz(), 9.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3Arithmetic)
{
  RealN3 a(1.0, 2.0, 3.0);
  RealN3 b(4.0, 5.0, 6.0);

  RealN3 c = a + b;
  ASSERT_EQ(c.vx(), 5.0);
  ASSERT_EQ(c.vy(), 7.0);
  ASSERT_EQ(c.vz(), 9.0);

  RealN3 d = c - b;
  ASSERT_EQ(d, a);

  RealN3 e = 2.0 * a;
  ASSERT_EQ(e.vx(), 2.0);
  ASSERT_EQ(e.vy(), 4.0);
  ASSERT_EQ(e.vz(), 6.0);

  RealN3 f = a * 2.0;
  ASSERT_EQ(f, e);

  RealN3 g = e / 2.0;
  ASSERT_EQ(g, a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, RealN3NotEqual)
{
  RealN3 v1(1.0, 2.0, 3.0);
  RealN3 v2(1.0, 2.0, 4.0);
  ASSERT_NE(v1, v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, Size4)
{
  using V4 = NumVector<Real, 4>;
  V4 v(1.0, 2.0, 3.0, 4.0);
  ASSERT_EQ(v[0], 1.0);
  ASSERT_EQ(v[1], 2.0);
  ASSERT_EQ(v[2], 3.0);
  ASSERT_EQ(v[3], 4.0);

  V4 w = v * 2.0;
  ASSERT_EQ(w[0], 2.0);
  ASSERT_EQ(w[3], 8.0);

  V4 z = V4::zero();
  ASSERT_TRUE(math::isNearlyZero(z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, Size5)
{
  using V5 = NumVector<Real, 5>;
  V5 v(1.0, 2.0, 3.0, 4.0, 5.0);
  ASSERT_EQ(v[0], 1.0);
  ASSERT_EQ(v[4], 5.0);

  v.fill(7.0);
  ASSERT_EQ(v[0], 7.0);
  ASSERT_EQ(v[4], 7.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, Size6)
{
  using V6 = NumVector<Real, 6>;
  V6 v(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  ASSERT_EQ(v[0], 1.0);
  ASSERT_EQ(v[3], 4.0);
  ASSERT_EQ(v[5], 6.0);

  V6 w;
  w = 3.0;
  ASSERT_EQ(w[0], 3.0);
  ASSERT_EQ(w[5], 3.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestNumVector, Int32x3)
{
  using V3i = NumVector<Int32, 3>;
  V3i v;
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 0);
  ASSERT_EQ(v[2], 0);

  v[0] = 1;
  v[1] = 2;
  v[2] = 3;
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[2], 3);

  V3i w(4, 5, 6);
  V3i s = v + w;
  ASSERT_EQ(s[0], 5);
  ASSERT_EQ(s[1], 7);
  ASSERT_EQ(s[2], 9);

  s *= 2;
  ASSERT_EQ(s[0], 10);
  ASSERT_EQ(s[1], 14);
  ASSERT_EQ(s[2], 18);

  struct A
  {
    Int64 x;
    Int32 z;
  };
  std::cout << "SIZEOF_A=" << sizeof(A) << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class NumVector<Real, 2>;
template class NumVector<Real, 3>;
template class NumVector<Real, 4>;
template class NumVector<Real, 5>;
template class NumVector<Real, 6>;
template class NumVector<Real, 1>;
template class NumVector<Int32, 3>;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
