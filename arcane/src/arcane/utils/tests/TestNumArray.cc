// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(NumArray,Basic)
{
  std::cout << "TEST_NUMARRAY Basic\n";

  NumArray<Real,1> array1(3);
  array1.s(1) = 5.0;
  ASSERT_EQ(array1(1),5.0);
  std::cout << " V=" << array1(1) << "\n";
  array1[2] = 3.0;
  ASSERT_EQ(array1[2],3.0);
  std::cout << " V=" << array1(1) << "\n";
  array1.resize(7);
  ASSERT_EQ(array1.totalNbElement(),7);

  NumArray<Real,2> array2(2,3);
  array2.s(1,2) = 5.0;
  std::cout << " V=" << array2(1,2) << "\n";
  array2.resize(7,5);
  ASSERT_EQ(array2.totalNbElement(),(7*5));

  NumArray<Real,3> array3(2,3,4);
  array3.s(1,2,3) = 5.0;
  std::cout << " V=" << array3(1,2,3) << "\n";
  array3.resize(12,4,6);
  ASSERT_EQ(array3.totalNbElement(),(12*4*6));

  NumArray<Real,4> array4(2,3,4,5);
  array4.s(1,2,3,4) = 5.0;
  std::cout << " V=" << array4(1,2,3,4) << "\n";
  array4.resize(8,3,7,5);
  ASSERT_EQ(array4.totalNbElement(),(8*3*7*5));

  NumArray<Real,1> num_data1(4, { 2.4, 5.6, 3.3, 5.4 });
  ASSERT_EQ(num_data1[0], 2.4);
  ASSERT_EQ(num_data1[1], 5.6);
  ASSERT_EQ(num_data1[2], 3.3);
  ASSERT_EQ(num_data1[3], 5.4);

  NumArray<Real,2,RightLayout<2>> num_data2(3, 2, { 1.4, 15.6, 33.3, 7.4, 4.2, 6.5 });
  ASSERT_EQ(num_data2(0,0), 1.4);
  ASSERT_EQ(num_data2(0,1), 15.6);
  ASSERT_EQ(num_data2(1,0), 33.3);
  ASSERT_EQ(num_data2(1,1), 7.4);
  ASSERT_EQ(num_data2(2,0), 4.2);
  ASSERT_EQ(num_data2(2,1), 6.5);
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
    for( Int32 x=0, xn=v.dim1Size(); x<xn; ++x ){
      for( Int32 y=0, yn=v.dim2Size(); y<yn; ++y ){
        for( Int32 z=0, zn=v.dim3Size(); z<zn; ++z ){
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

  // NOTE: désactive temporairement le test tant que la méthode
  // resize() de 'NumArray' ne conserve pas les valeurs
#if NUMARRAY_HAS_VALID_RESIZE
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
#endif
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

  {
    NumArray<int,1> vi0(4,{1,3,5,7});
    NumArray<int,1> vi1(vi0);
    NumArray<int,1> vi2;
    vi2 = vi1;
    ASSERT_EQ(vi1.to1DSpan(),vi0.to1DSpan());
    ASSERT_EQ(vi2.to1DSpan(),vi1.to1DSpan());
    NumArray<int,1> vi3(vi0.to1DSpan());
    ASSERT_EQ(vi3.to1DSpan(),vi0.to1DSpan());
  }

  {
    NumArray<int,1> vi0(4,{1,3,5,7});
    NumArray<int,1> vi1(vi0);
    NumArray<int,1> vi2;
    vi2 = vi1;
    ASSERT_EQ(vi1.to1DSpan(),vi0.to1DSpan());
    ASSERT_EQ(vi2.to1DSpan(),vi1.to1DSpan());
  }
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

namespace
{
template<typename T>
void _setNumArray2Values(T& a)
{
  for( Int32 i=0; i<a.dim1Size(); ++i ){
    for( Int32 j=0; j<a.dim2Size(); ++j ){
      a.s(i,j) = (i*253) + j;
    }
  }
}
template<typename T>
void _setNumArray3Values(T& a)
{
  for( Int32 i=0; i<a.dim1Size(); ++i ){
    for( Int32 j=0; j<a.dim2Size(); ++j ){
      for( Int32 k=0; k<a.dim3Size(); ++k ){
        a.s(i,j,k) = (i*253) + (j*27) + k;
      }
    }
  }
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray2,Layout)
{
  std::cout << "TEST_NUMARRAY2 Layout\n";

  {
    NumArray<Real,2,RightLayout2> a(3,5);
    ASSERT_EQ(a.totalNbElement(),(3*5));
    _setNumArray2Values(a);
    auto values = a.to1DSpan();
    std::cout << "V=" << values << "\n";
    UniqueArray<Real> ref_value = { 0, 1, 2, 3, 4, 253, 254, 255, 256, 257, 506, 507, 508, 509, 510 };
    ASSERT_EQ(values.smallView(),ref_value.view());
  }

  {
    NumArray<Real,2,LeftLayout2> a(3,5);
    ASSERT_EQ(a.totalNbElement(),(3*5));
    _setNumArray2Values(a);
    auto values = a.to1DSpan();
    std::cout << "V=" << values << "\n";
    UniqueArray<Real> ref_value = { 0, 253, 506, 1, 254, 507, 2, 255, 508, 3, 256, 509, 4, 257, 510 };
    ASSERT_EQ(values.smallView(),ref_value.view());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray3,Layout)
{
  std::cout << "TEST_NUMARRAY3 Layout\n";

  {
    NumArray<Real,3,RightLayout3> a(2,3,5);
    ASSERT_EQ(a.totalNbElement(),(2*3*5));
    _setNumArray3Values(a);
    auto values = a.to1DSpan();
    std::cout << "V=" << values << "\n";
    UniqueArray<Real> ref_value =
    {
      0, 1, 2, 3, 4, 27, 28, 29, 30, 31, 54, 55, 56, 57, 58,
      253, 254, 255, 256, 257, 280, 281, 282, 283, 284, 307, 308, 309, 310, 311
    };
    ASSERT_EQ(values.smallView(),ref_value.view());
  }

  {
    NumArray<Real,3,LeftLayout3> a(2,3,5);
    ASSERT_EQ(a.totalNbElement(),(2*3*5));
    _setNumArray3Values(a);
    auto values = a.to1DSpan();
    std::cout << "V=" << values << "\n";
    UniqueArray<Real> ref_value =
    {
      0, 253, 27, 280, 54, 307, 1, 254, 28, 281, 55, 308, 2, 255, 29,
      282, 56, 309, 3, 256, 30, 283, 57, 310, 4, 257, 31, 284, 58, 311
    };
    ASSERT_EQ(values.smallView(),ref_value.view());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray,RealN)
{
  {
    NumArray<Real2,1> a(5);
    a(2) = Real2(0.0,3.2);
    a(3,1) = 2.0;
    ASSERT_EQ(a(3).y,2.0);
  }

  {
    NumArray<Real3,1> a(5);
    const Real3 v(0.0,3.2,5.6);
    a(0) = v;
    a(4,1) = 4.0;
    ASSERT_EQ(a(4).y,4.0);
    ASSERT_EQ(a(0),v);
  }

  {
    NumArray<Real2x2,1> a(5);
    const Real2 v0(1.2,1.7);
    const Real2x2 v(Real2(3.2,5.6), Real2(3.4,1.7));
    a(0) = v;
    a(3,1,0) = v0.x;
    a(3,1,1) = v0.y;
    a(4,1,0) = 4.0;
    ASSERT_EQ(a(4).y.x,4.0);
    ASSERT_EQ(a(0),v);
    ASSERT_EQ(a(3,1),v0);
  }

  {
    NumArray<Real3x3,1> a(5);
    const Real3 v0(1.2,3.4,1.7);
    const Real3x3 v(Real3(0.0,3.2,5.6), Real3(1.2,3.4,1.7), Real3(9.2,1.4,5.0));
    a(0) = v;
    a(3,1,0) = v0.x;
    a(3,1,1) = v0.y;
    a(3,1,2) = v0.z;
    a(4,1,2) = 4.0;
    ASSERT_EQ(a(4).y.z,4.0);
    ASSERT_EQ(a(0),v);
    ASSERT_EQ(a(3,1),v0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class NumArray<float,4,RightLayout<4>>;
template class NumArray<float,3,RightLayout<3>>;
template class NumArray<float,2,RightLayout<2>>;

template class NumArray<float,4,LeftLayout<4>>;
template class NumArray<float,3,LeftLayout<3>>;
template class NumArray<float,2,LeftLayout<2>>;

template class NumArray<float,1>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
