// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NumericTypes.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(TestRealN,Real2)
{
  {
    Real2 v1;
    ASSERT_EQ(v1.x,0.0);
    ASSERT_EQ(v1.y,0.0);
  }
  {
    double value = 0.2;
    Real2 v1(value);
    ASSERT_EQ(v1.x,value);
    ASSERT_EQ(v1.y,value);
    Real2 v2(v1);
    ASSERT_EQ(v2.x,value);
    ASSERT_EQ(v2.y,value);
    v2.x = 3.5;
    v2.y = 1.2;
    v1 = v2;
    ASSERT_EQ(v1.x,3.5);
    ASSERT_EQ(v1.y,1.2);
    ASSERT_EQ(v1,v2);
  }
  {
    double value = 0.3;
    Real2 v2;
    v2 = value;
    ASSERT_EQ(v2.x,value);
    ASSERT_EQ(v2.x,value);
  }
  // Operator + and -
  {
    Real2 v2(1.2,4.5);
    Real2 v1(1.3,2.3);
    Real2 v3 = v1 + v2;
    ASSERT_EQ(v3.x,2.5);
    ASSERT_EQ(v3.y,6.8);
    Real2 v4 = v3 - v2;
    ASSERT_EQ(v4,v1);
  }
  // Operator * and /
  {
    Real2 v2(1.2,4.5);
    Real2 v1(1.3,2.3);
    Real2 v3 = v1 * v2;
    ASSERT_EQ(v3.x,1.2*1.3);
    ASSERT_EQ(v3.y,4.5*2.3);
    Real2 v4 = v3 * 2.3;
    ASSERT_EQ(v4.x,v3.x*2.3);
    ASSERT_EQ(v4.y,v3.y*2.3);
    Real2 v5 = v4 / 2.4;
    ASSERT_EQ(v5.x,v4.x/2.4);
    ASSERT_EQ(v5.y,v4.y/2.4);
  }
  {
    Real3 v3(1.2,2.3,4.5);
    Real2 v2(v3);
    ASSERT_EQ(v2.x,1.2);
    ASSERT_EQ(v2.y,2.3);
  }
}

TEST(TestRealN,Real3)
{
  {
    Real3 v1;
    ASSERT_EQ(v1.x,0.0);
    ASSERT_EQ(v1.y,0.0);
    ASSERT_EQ(v1.z,0.0);
  }
  {
    double value = 0.2;
    Real3 v1(value);
    ASSERT_EQ(v1.x,value);
    ASSERT_EQ(v1.y,value);
    ASSERT_EQ(v1.z,value);
    Real3 v2(v1);
    ASSERT_EQ(v2.x,value);
    ASSERT_EQ(v2.y,value);
    ASSERT_EQ(v2.z,value);
    v2.x = 3.5;
    v2.y = 1.2;
    v2.z = -1.5;
    v1 = v2;
    ASSERT_EQ(v1.x,3.5);
    ASSERT_EQ(v1.y,1.2);
    ASSERT_EQ(v1.z,-1.5);
    ASSERT_EQ(v1,v2);
  }
  {
    Real2 v2(1.2,2.3);
    Real3 v3(v2);
    ASSERT_EQ(v3.x,1.2);
    ASSERT_EQ(v3.y,2.3);
    ASSERT_EQ(v3.z,0.0);
  }
}

TEST(TestRealN,Real2x2)
{
  Real2 zero;
  {
    Real2x2 v1;
    ASSERT_EQ(v1.x,zero);
    ASSERT_EQ(v1.y,zero);
  }
  {
    double value = 0.2;
    Real2 r2_value(value);
    Real2x2 v1(value);
    ASSERT_EQ(v1.x,r2_value);
    ASSERT_EQ(v1.y,r2_value);
    Real2x2 v2(v1);
    ASSERT_EQ(v2.x,v1.x);
    ASSERT_EQ(v2.y,v1.y);
    Real2 rx(3.5, 1.2);
    Real2 ry(1.6, 2.1);
    v2.x = rx;
    v2.y = ry;
    v1 = v2;
    ASSERT_EQ(v1.x,rx);
    ASSERT_EQ(v1.y,ry);
    ASSERT_EQ(v1,v2);
  }
}

TEST(TestRealN,Real3x3)
{
  Real3 zero;
  {
    Real3x3 v1;
    ASSERT_EQ(v1.x,zero);
    ASSERT_EQ(v1.y,zero);
    ASSERT_EQ(v1.z,zero);
  }
  {
    double value = 0.2;
    Real3 r3_value(value);
    Real3x3 v1(value);
    ASSERT_EQ(v1.x,r3_value);
    ASSERT_EQ(v1.y,r3_value);
    ASSERT_EQ(v1.z,r3_value);
    Real3x3 v2(v1);
    ASSERT_EQ(v2.x,v1.x);
    ASSERT_EQ(v2.y,v1.y);
    ASSERT_EQ(v2.z,v1.z);
    Real3 rx(3.5, 1.2, -1.5);
    Real3 ry(1.6, 2.1, -2.3);
    Real3 rz(-2.3, 1.8, 9.4);
    v2.x = rx;
    v2.y = ry;
    v2.z = rz;
    v1 = v2;
    ASSERT_EQ(v1.x,rx);
    ASSERT_EQ(v1.y,ry);
    ASSERT_EQ(v1.z,rz);
    ASSERT_EQ(v1,v2);
  }
}

TEST(TestRealN,Copyable)
{
  ASSERT_TRUE(std::is_trivially_copyable_v<Real2>);
  ASSERT_TRUE(std::is_trivially_copyable_v<Real3>);
  ASSERT_TRUE(std::is_trivially_copyable_v<Real2x2>);
  ASSERT_TRUE(std::is_trivially_copyable_v<Real3x3>);
}

TEST(TestRealN, ContiguousAddress)
{
  // Vérifie que les adresses des champs des classes 'RealN' sont
  // contigues et qu'on peut utiliser operator[] pour accéder à la i-ème composante.
  Real2 r2;
  ASSERT_TRUE(sizeof(r2) == 2 * sizeof(Real));
  ASSERT_TRUE(&r2.y == (&r2.x + 1));

  Real3 r3;
  ASSERT_TRUE(sizeof(r3) == 3 * sizeof(Real));
  Real* r3_base = &r3.x;
  ASSERT_TRUE(&r3.y == (r3_base + 1));
  ASSERT_TRUE(&r3.z == (r3_base + 2));

  Real2x2 r2x2;
  Real2* r2x2_base = &r2x2.x;
  Real* r2x2_base_x = &r2x2.x.x;
  Real* r2x2_base_y = &r2x2.y.x;

  ASSERT_TRUE(sizeof(r2x2) == 4 * sizeof(Real));

  ASSERT_TRUE(&r2x2.x == r2x2_base + 0);
  ASSERT_TRUE(&r2x2.y == r2x2_base + 1);

  ASSERT_EQ(static_cast<void*>(r2x2_base_x), r2x2_base + 0);
  ASSERT_EQ(static_cast<void*>(r2x2_base_y), r2x2_base + 1);

  ASSERT_TRUE(&r2x2.x.x == r2x2_base_x + 0);
  ASSERT_TRUE(&r2x2.x.y == r2x2_base_x + 1);

  ASSERT_TRUE(&r2x2.y.x == r2x2_base_y + 0);
  ASSERT_TRUE(&r2x2.y.y == r2x2_base_y + 1);

  Real3x3 r3x3;
  Real3* r3x3_base = &r3x3.x;
  Real* r3x3_base_x = &r3x3.x.x;
  Real* r3x3_base_y = &r3x3.y.x;
  Real* r3x3_base_z = &r3x3.z.x;
  ASSERT_TRUE(sizeof(r3x3) == 9 * sizeof(Real));

  ASSERT_TRUE(&r3x3.x == r3x3_base + 0);
  ASSERT_TRUE(&r3x3.y == r3x3_base + 1);
  ASSERT_TRUE(&r3x3.z == r3x3_base + 2);

  ASSERT_EQ(static_cast<void*>(r3x3_base_x), r3x3_base + 0);
  ASSERT_EQ(static_cast<void*>(r3x3_base_y), r3x3_base + 1);
  ASSERT_EQ(static_cast<void*>(r3x3_base_z), r3x3_base + 2);

  ASSERT_TRUE(&r3x3.x.x == r3x3_base_x + 0);
  ASSERT_TRUE(&r3x3.x.y == r3x3_base_x + 1);
  ASSERT_TRUE(&r3x3.x.z == r3x3_base_x + 2);

  ASSERT_TRUE(&r3x3.y.x == r3x3_base_y + 0);
  ASSERT_TRUE(&r3x3.y.y == r3x3_base_y + 1);
  ASSERT_TRUE(&r3x3.y.z == r3x3_base_y + 2);

  ASSERT_TRUE(&r3x3.z.x == r3x3_base_z + 0);
  ASSERT_TRUE(&r3x3.z.y == r3x3_base_z + 1);
  ASSERT_TRUE(&r3x3.z.z == r3x3_base_z + 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
