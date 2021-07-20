// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(NumArray,Basic)
{
  std::cout << "TEST_NUMARRAY Basic\n";

  NumArray<Real,1> array1(2);
  array1.s(1) = 5.0;
  std::cout << " V=" << array1(1) << "\n";
  array1.resize(7);

  NumArray<Real,2> array2(2,3);
  array2.s(1,2) = 5.0;
  std::cout << " V=" << array2(1,2) << "\n";
  array2.resize(7,5);

  NumArray<Real,3> array3(2,3,4);
  array3.s(1,2,3) = 5.0;
  std::cout << " V=" << array3(1,2,3) << "\n";
  array3.resize(12,4,6);

  NumArray<Real,4> array4(2,3,4,5);
  array4.s(1,2,3,4) = 5.0;
  std::cout << " V=" << array4(1,2,3,4) << "\n";
  array4.resize(8,3,7,5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray,Basic2)
{
  std::cout << "TEST_NUMARRAY Basic2\n";

  NumArray<Real,1> array1;
  array1.resize(2);
  array1.s(1) = 5.0;
  std::cout << " V=" << array1(1) << "\n";

  NumArray<Real,2> array2;
  array2.resize(2,3);
  array2.s(1,2) = 5.0;
  std::cout << " V=" << array2(1,2) << "\n";

  NumArray<Real,3> array3(2,3,4);
  array3.resize(2,3,4);
  array3.s(1,2,3) = 5.0;
  std::cout << " V=" << array3(1,2,3) << "\n";

  NumArray<Real,4> array4(2,3,4,5);
  array4.resize(2,3,4,5);
  array4.s(1,2,3,4) = 5.0;
  std::cout << " V=" << array4(1,2,3,4) << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray3,Misc)
{
  int nb_x = 3;
  int nb_y = 4;
  int nb_z = 5;

  NumArray<Int64,3> v(nb_x,nb_y,nb_z);
  {
    for( Int64 x=0, xn=v.dim1Size(); x<xn; ++x ){
      for( Int64 y=0, yn=v.dim2Size(); y<yn; ++y ){
        for( Int64 z=0, zn=v.dim3Size(); z<zn; ++z ){
          v.s(x,y,z) = x+y+z+1;
          v.s({x,y,z}) = x+y+z+1;
        }
      }
    }
  }

  for( Int64 x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Int64 y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Int64 z=0, zn=v.dim3Size(); z<zn; ++z ){
        Int64 val1 = v(x,y,z);
        Int64 val2 = v({x,y,z});
        ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
      }
    }
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray3,Copy)
{
  int nb_x = 3;
  int nb_y = 4;
  int nb_z = 5;
  NumArray<Real,3> v(nb_x,nb_y,nb_z);
  v.fill(3.2);
  NumArray<Real,3> v2(nb_x*2,nb_y/2,nb_z*3);

  v.copy(v2.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class NumArray<float,4>;
template class NumArray<float,3>;
template class NumArray<float,2>;
template class NumArray<float,1>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
