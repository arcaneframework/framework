// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"

#include <vector>

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
  constexpr int nb_x = 3;
  constexpr int nb_y = 4;
  constexpr int nb_z = 5;

  NumArray<Int64,3> v(nb_x,nb_y,nb_z);
  v.fill(0);
  // Attention, v.extents() change si 'v' est redimensionné
  auto v_extents = v.extentsWithOffset();
  {
    for( Int64 x=0, xn=v.dim1Size(); x<xn; ++x ){
      for( Int64 y=0, yn=v.dim2Size(); y<yn; ++y ){
        for( Int64 z=0, zn=v.dim3Size(); z<zn; ++z ){
          ArrayBoundsIndex<3> idx{x,y,z};
          Int64 offset = v_extents.offset(idx);
          v.s(x,y,z) = offset;
          v.s({x,y,z}) = offset;
          v.s(idx) = offset;
        }
      }
    }
  }
  std::cout << "CAPACITY V1=" << v.capacity() << "\n";
  v.resize(4,5,6);
  std::cout << "CAPACITY V2=" << v.capacity() << "\n";
  v.resize(2,7,9);
  std::cout << "CAPACITY V3=" << v.capacity() << "\n";
  v.resize(3,2,6);
  std::cout << "CAPACITY V4=" << v.capacity() << "\n";

  // Les valeurs ci-dessous dépendent de l'implémentation actuelle
  // de NumArray::resize(). Je ne suis pas sur qu'il soit valide de les
  // tester
  std::vector<Int64> valid_values =
  {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    24 , 25, 26, 27, 28, 29, 20 , 21, 22, 23, 24, 25
  };
  ASSERT_EQ(valid_values.size(),(size_t)36);

  v_extents = v.extentsWithOffset();
  Int64 index = 0;
  for( Int64 x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Int64 y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Int64 z=0, zn=v.dim3Size(); z<zn; ++z ){
        ArrayBoundsIndex<3> idx{x,y,z};
        Int64 offset = v_extents.offset(idx);
        Int64 val1 = v(x,y,z);
        Int64 val2 = v({x,y,z});
        std::cout << "XYZ=" << x << " " << y << " " << z
                  << " V=" << val1 << " offset=" << offset << "\n";
        ASSERT_EQ(index,offset);
        ASSERT_EQ(val1,val2);
        ASSERT_EQ(valid_values.at(offset),val1);
        ++index;
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

TEST(NumArray3,Index)
{
  ArrayBoundsIndex<3> index(1,4,2);
  auto [i, j, k] = index();

  ASSERT_TRUE(i==1);
  ASSERT_TRUE(j==4);
  ASSERT_TRUE(k==2);
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
