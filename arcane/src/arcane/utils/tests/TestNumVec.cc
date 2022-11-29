// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NumVec.h"
#include "arcane/utils/NumMat.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

using RealN2 = NumVec<Real,2>;
using RealN3 = NumVec<Real,3>;
using RealN2x2 = NumMat<Real,2>;
using RealN3x3 = NumMat<Real,3>;

TEST(TestNumVec,RealN2)
{
  std::cout << "   sizeof(NumVec<double,2>) = " << sizeof(NumVec<double,2>) << "\n";
  std::cout << "   sizeof(NumVec<double,3>) = " << sizeof(NumVec<double,3>) << "\n";
  std::cout << "   sizeof(NumVec<float,2>) = " << sizeof(NumVec<float,2>) << "\n";
  std::cout << "   sizeof(NumVec<float,3>) = " << sizeof(NumVec<float,3>) << "\n";
  std::cout << "   sizeof(NumVec<short,2>) = " << sizeof(NumVec<short,2>) << "\n";
  std::cout << "   sizeof(NumVec<short,3>) = " << sizeof(NumVec<short,3>) << "\n";
  {
    RealN2 v1;
    ASSERT_EQ(v1.x(),0.0);
    ASSERT_EQ(v1.y(),0.0);
  }
  {
    double value = 0.2;
    RealN2 v1(value);
    ASSERT_EQ(v1.x(),value);
    ASSERT_EQ(v1.y(),value);
    RealN2 v2(v1);
    ASSERT_EQ(v2.x(),value);
    ASSERT_EQ(v2.y(),value);
    v2.x() = 3.5;
    v2.y() = 1.2;
    v1 = v2;
    ASSERT_EQ(v1.x(),3.5);
    ASSERT_EQ(v1.y(),1.2);
    ASSERT_EQ(v1,v2);
  }
  {
    double value = 0.3;
    RealN2 v2;
    v2 = value;
    ASSERT_EQ(v2.x(),value);
    ASSERT_EQ(v2.x(),value);
  }
  // Operator + and -
  {
    RealN2 v2(1.2,4.5);
    RealN2 v1(1.3,2.3);
    RealN2 v3 = v1 + v2;
    ASSERT_EQ(v3.x(),2.5);
    ASSERT_EQ(v3.y(),6.8);
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
    ASSERT_EQ(v4.x(),v2.x()*2.3);
    ASSERT_EQ(v4.y(),v2.y()*2.3);
    RealN2 v5 = v4 / 2.4;
    ASSERT_EQ(v5.x(),v4.x()/2.4);
    ASSERT_EQ(v5.y(),v4.y()/2.4);
  }
}

TEST(TestNumVec,Real3)
{
  {
    RealN3 v1;
    ASSERT_EQ(v1.x(),0.0);
    ASSERT_EQ(v1.y(),0.0);
    ASSERT_EQ(v1.z(),0.0);
  }
  {
    double value = 0.2;
    RealN3 v1(value);
    ASSERT_EQ(v1.x(),value);
    ASSERT_EQ(v1.y(),value);
    ASSERT_EQ(v1.z(),value);
    RealN3 v2(v1);
    ASSERT_EQ(v2.x(),value);
    ASSERT_EQ(v2.y(),value);
    ASSERT_EQ(v2.z(),value);
    v2.x() = 3.5;
    v2.y() = 1.2;
    v2.z() = -1.5;
    v1 = v2;
    ASSERT_EQ(v1.x(),3.5);
    ASSERT_EQ(v1.y(),1.2);
    ASSERT_EQ(v1.z(),-1.5);
    ASSERT_EQ(v1,v2);
  }
}

TEST(TestNumMat,Real2x2)
{
  RealN2 zero;
  {
    RealN2x2 v1;
    ASSERT_EQ(v1.x(),zero);
    ASSERT_EQ(v1.y(),zero);
  }
  {
    double value = 0.2;
    RealN2 r2_value(value);
    RealN2x2 v1(value);
    ASSERT_EQ(v1.x(),r2_value);
    ASSERT_EQ(v1.y(),r2_value);
    RealN2x2 v2(v1);
    ASSERT_EQ(v2.x(),v1.x());
    ASSERT_EQ(v2.y(),v1.y());
    RealN2 rx(3.5, 1.2);
    RealN2 ry(1.6, 2.1);
    v2.x() = rx;
    v2.y() = ry;
    v1 = v2;
    ASSERT_EQ(v1.x(),rx);
    ASSERT_EQ(v1.y(),ry);
    ASSERT_EQ(v1,v2);
  }
}

TEST(TestNumVec,Real3x3)
{
  RealN3 zero;
  {
    RealN3x3 v1;
    ASSERT_EQ(v1.x(),zero);
    ASSERT_EQ(v1.y(),zero);
    ASSERT_EQ(v1.z(),zero);
  }
  {
    double value = 0.2;
    RealN3 r3_value(value);
    RealN3x3 v1(value);
    ASSERT_EQ(v1.x(),r3_value);
    ASSERT_EQ(v1.y(),r3_value);
    ASSERT_EQ(v1.z(),r3_value);
    RealN3x3 v2(v1);
    ASSERT_EQ(v2.x(),v1.x());
    ASSERT_EQ(v2.y(),v1.y());
    ASSERT_EQ(v2.z(),v1.z());
    RealN3 rx(3.5, 1.2, -1.5);
    RealN3 ry(1.6, 2.1, -2.3);
    RealN3 rz(-2.3, 1.8, 9.4);
    v2.x() = rx;
    v2.y() = ry;
    v2.z() = rz;
    v1 = v2;
    ASSERT_EQ(v1.x(),rx);
    ASSERT_EQ(v1.y(),ry);
    ASSERT_EQ(v1.z(),rz);
    ASSERT_EQ(v1,v2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class NumVec<Real,2>;
template class NumVec<Real,3>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
