// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NumVector.h"
#include "arcane/utils/NumMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(TestNumVector,RealN2)
{
  std::cout << "   sizeof(NumVector<double,2>) = " << sizeof(NumVector<double,2>) << "\n";
  std::cout << "   sizeof(NumVector<double,3>) = " << sizeof(NumVector<double,3>) << "\n";
  std::cout << "   sizeof(NumVector<float,2>) = " << sizeof(NumVector<float,2>) << "\n";
  std::cout << "   sizeof(NumVector<float,3>) = " << sizeof(NumVector<float,3>) << "\n";
  std::cout << "   sizeof(NumVector<short,2>) = " << sizeof(NumVector<short,2>) << "\n";
  std::cout << "   sizeof(NumVector<short,3>) = " << sizeof(NumVector<short,3>) << "\n";
  {
    RealN2 v1;
    ASSERT_EQ(v1.vx(),0.0);
    ASSERT_EQ(v1.vy(),0.0);
  }
  {
    double value = 0.2;
    RealN2 v1(value);
    ASSERT_EQ(v1.vx(),value);
    ASSERT_EQ(v1.vy(),value);
    RealN2 v2(v1);
    ASSERT_EQ(v2.vx(),value);
    ASSERT_EQ(v2.vy(),value);
    v2.vx() = 3.5;
    v2.vy() = 1.2;
    v1 = v2;
    ASSERT_EQ(v1.vx(),3.5);
    ASSERT_EQ(v1.vy(),1.2);
    ASSERT_EQ(v1,v2);
  }
  {
    double value = 0.3;
    RealN2 v2;
    v2 = value;
    ASSERT_EQ(v2.vx(),value);
    ASSERT_EQ(v2.vx(),value);
  }
  // Operator + and -
  {
    RealN2 v2(1.2,4.5);
    RealN2 v1(1.3,2.3);
    RealN2 v3 = v1 + v2;
    ASSERT_EQ(v3.vx(),2.5);
    ASSERT_EQ(v3.vy(),6.8);
    RealN2 v4 = v3 - v2;
    ASSERT_EQ(v4,v1);
  }
  // Operator * and /
  {
    RealN2 v2(1.2,4.5);
    RealN2 v1(1.3,2.3);
    //RealN2 v3 = v1 * v2;
    //ASSERT_EQ(v3.x(),1.2*1.3);
    //ASSERT_EQ(v3.y(),4.5*2.3);
    RealN2 v4 = v2 * 2.3;
    ASSERT_EQ(v4.vx(),v2.vx()*2.3);
    ASSERT_EQ(v4.vy(),v2.vy()*2.3);
    RealN2 v5 = v4 / 2.4;
    ASSERT_EQ(v5.vx(),v4.vx()/2.4);
    ASSERT_EQ(v5.vy(),v4.vy()/2.4);
  }
}

TEST(TestNumVector,Real3)
{
  {
    RealN3 v1;
    ASSERT_EQ(v1.vx(),0.0);
    ASSERT_EQ(v1.vy(),0.0);
    ASSERT_EQ(v1.vz(),0.0);
  }
  {
    double value = 0.2;
    RealN3 v1(value);
    ASSERT_EQ(v1.vx(),value);
    ASSERT_EQ(v1.vy(),value);
    ASSERT_EQ(v1.vz(),value);
    RealN3 v2(v1);
    ASSERT_EQ(v2.vx(),value);
    ASSERT_EQ(v2.vy(),value);
    ASSERT_EQ(v2.vz(),value);
    v2.vx() = 3.5;
    v2.vy() = 1.2;
    v2.vz() = -1.5;
    v1 = v2;
    ASSERT_EQ(v1.vx(),3.5);
    ASSERT_EQ(v1.vy(),1.2);
    ASSERT_EQ(v1.vz(),-1.5);
    ASSERT_EQ(v1,v2);
  }
}

TEST(TestNumMat,Real2x2)
{
  RealN2 zero;
  {
    RealN2x2 v1;
    ASSERT_EQ(v1.vx(),zero);
    ASSERT_EQ(v1.vy(),zero);
  }
  {
    double value = 0.2;
    RealN2 r2_value(value);
    RealN2x2 v1(value);
    ASSERT_EQ(v1.vx(),r2_value);
    ASSERT_EQ(v1.vy(),r2_value);
    RealN2x2 v2(v1);
    ASSERT_EQ(v2.vx(),v1.vx());
    ASSERT_EQ(v2.vy(),v1.vy());
    RealN2 rx(3.5, 1.2);
    RealN2 ry(1.6, 2.1);
    v2.vx() = rx;
    v2.vy() = ry;
    v1 = v2;
    ASSERT_EQ(v1.vx(),rx);
    ASSERT_EQ(v1.vy(),ry);
    ASSERT_EQ(v1,v2);
  }
}

TEST(TestNumVector,Real3x3)
{
  RealN3 zero;
  {
    RealN3x3 v1;
    ASSERT_EQ(v1.vx(),zero);
    ASSERT_EQ(v1.vy(),zero);
    ASSERT_EQ(v1.vz(),zero);
  }
  {
    double value = 0.2;
    RealN3 r3_value(value);
    RealN3x3 v1(value);
    ASSERT_EQ(v1.vx(),r3_value);
    ASSERT_EQ(v1.vy(),r3_value);
    ASSERT_EQ(v1.vz(),r3_value);
    RealN3x3 v2(v1);
    ASSERT_EQ(v2.vx(),v1.vx());
    ASSERT_EQ(v2.vy(),v1.vy());
    ASSERT_EQ(v2.vz(),v1.vz());
    RealN3 rx(3.5, 1.2, -1.5);
    RealN3 ry(1.6, 2.1, -2.3);
    RealN3 rz(-2.3, 1.8, 9.4);
    v2.vx() = rx;
    v2.vy() = ry;
    v2.vz() = rz;
    v1 = v2;
    ASSERT_EQ(v1.vx(),rx);
    ASSERT_EQ(v1.vy(),ry);
    ASSERT_EQ(v1.vz(),rz);
    ASSERT_EQ(v1,v2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class NumVector<Real,2>;
template class NumVector<Real,3>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
