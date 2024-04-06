// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/utils/NumArrayUtils.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(NumArray, Basic)
{
  std::cout << "TEST_NUMARRAY Basic\n";

  NumArray<Real, MDDim1> array1(3);
  array1(1) = 5.0;
  ASSERT_EQ(array1(1), 5.0);
  std::cout << " V=" << array1(1) << "\n";
  array1[2] = 3.0;
  ASSERT_EQ(array1[2], 3.0);
  std::cout << " V=" << array1(1) << "\n";
  array1.resize(7);
  ASSERT_EQ(array1.totalNbElement(), 7);

  NumArray<Real, MDDim2> array2(2, 3);
  array2(1, 2) = 5.0;
  std::cout << " V=" << array2(1, 2) << "\n";
  array2.resize(7, 5);
  ASSERT_EQ(array2.totalNbElement(), (7 * 5));

  NumArray<Real, MDDim3> array3(2, 3, 4);
  array3(1, 2, 3) = 5.0;
  std::cout << " V=" << array3(1, 2, 3) << "\n";
  ASSERT_EQ(array3(1, 2, 3), 5.0);
  array3.resize(12, 4, 6);
  ASSERT_EQ(array3.totalNbElement(), (12 * 4 * 6));
  array3.fill(0.0);
  array3(1, 2, 3) = 4.0;
  array3(2, 3, 5) = 1.0;

  {
    MDSpan<Real, MDDim3> span_array3(array3.mdspan());
    ASSERT_EQ(array3.extent0(), span_array3.extent0());

    MDSpan<const Real, MDDim3> const_span_array3(array3.constMDSpan());
    ASSERT_EQ(const_span_array3.to1DSpan(), span_array3.to1DSpan());

    ASSERT_EQ(array3.extent0(), span_array3.extent0());
    std::cout << "Array3: extents=" << array3.extent0()
              << "," << array3.extent1() << "," << array3.extent2() << "\n";
    for (Int32 i = 0; i < array3.extent0(); ++i) {
      MDSpan<Real, MDDim2> span_array2 = span_array3.slice(i);
      ASSERT_EQ(span_array2.extent0(), span_array3.extent1());
      ASSERT_EQ(span_array2.extent1(), span_array3.extent2());
      std::cout << " MDDim2 slice i=" << i << " X=" << span_array2.extent0() << " Y=" << span_array2.extent1() << "\n";
      for (Int32 x = 0, xn = span_array2.extent0(); x < xn; ++x) {
        for (Int32 y = 0, yn = span_array2.extent1(); y < yn; ++y) {
          ASSERT_EQ(span_array2.ptrAt(x, y), span_array3.ptrAt(i, x, y));
        }
      }
    }
  }
  {
    MDSpan<Real, MDDim2> span_array2(array2.mdspan());
    std::cout << "Array2: extents=" << array2.extent0() << "," << array2.extent1() << "\n";
    for (Int32 i = 0; i < array2.extent0(); ++i) {
      MDSpan<Real, MDDim1> span_array1 = array2.span().slice(i);
      ASSERT_EQ(span_array1.extent0(), span_array2.extent1());
      std::cout << " MDDim1 slice i=" << i << " X=" << span_array2.extent0() << "\n";
      for (Int32 x = 0, xn = span_array1.extent0(); x < xn; ++x) {
        ASSERT_EQ(span_array1.ptrAt(x), span_array2.ptrAt(i, x));
      }
    }
  }
  NumArray<Real, MDDim4> array4(2, 3, 4, 5);
  array4(1, 2, 3, 4) = 5.0;
  std::cout << " V=" << array4(1, 2, 3, 4) << "\n";
  array4.resize(8, 3, 7, 5);
  ASSERT_EQ(array4.totalNbElement(), (8 * 3 * 7 * 5));

  NumArray<Real, MDDim1> num_data1(4, { 2.4, 5.6, 3.3, 5.4 });
  ASSERT_EQ(num_data1[0], 2.4);
  ASSERT_EQ(num_data1[1], 5.6);
  ASSERT_EQ(num_data1[2], 3.3);
  ASSERT_EQ(num_data1[3], 5.4);

  NumArray<Real, MDDim2, RightLayout> num_data2(3, 2, { 1.4, 15.6, 33.3, 7.4, 4.2, 6.5 });
  ASSERT_EQ(num_data2(0, 0), 1.4);
  ASSERT_EQ(num_data2(0, 1), 15.6);
  ASSERT_EQ(num_data2(1, 0), 33.3);
  ASSERT_EQ(num_data2(1, 1), 7.4);
  ASSERT_EQ(num_data2(2, 0), 4.2);
  ASSERT_EQ(num_data2(2, 1), 6.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray,Basic2)
{
  std::cout << "TEST_NUMARRAY Basic2\n";

  NumArray<Real,MDDim1> array1;
  array1.resize(2);
  array1(1) = 5.0;
  std::cout << " V=" << array1(1) << "\n";

  NumArray<Real,MDDim2> array2;
  array2.resize(2,3);
  array2(1,2) = 5.0;
  std::cout << " V=" << array2(1,2) << "\n";

  NumArray<Real,MDDim3> array3(2,3,4);
  array3.resize(2,3,4);
  array3(1,2,3) = 5.0;
  std::cout << " V=" << array3(1,2,3) << "\n";

  NumArray<Real,MDDim4> array4(2,3,4,5);
  array4.resize(2,3,4,5);
  array4(1,2,3,4) = 5.0;
  std::cout << " V=" << array4(1,2,3,4) << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray,Extents)
{
  std::cout << "TEST_NUMARRAY Extents\n";

  ASSERT_EQ(1,(ExtentsV<-1,1,1,1>::nb_dynamic));
  ASSERT_EQ(2,(ExtentsV<-1,DynExtent,2>::nb_dynamic));
  ASSERT_EQ(1,(ExtentsV<DynExtent,1>::nb_dynamic));
  ASSERT_EQ(1,(ExtentsV<-1>::nb_dynamic));
  ASSERT_EQ(2,(ExtentsV<DynExtent,DynExtent,1>::nb_dynamic));
  ASSERT_EQ(1,MDDim1::nb_dynamic);
  ASSERT_EQ(2,MDDim2::nb_dynamic);
  ASSERT_EQ(3,MDDim3::nb_dynamic);;
  ASSERT_EQ(4,MDDim4::nb_dynamic);

  {
    NumArray<int,ExtentsV<2,DynExtent,3,DynExtent>> x1;
    x1.resize({6,7});
    ASSERT_EQ(x1.extent0(),2);
    ASSERT_EQ(x1.extent1(),6);
    ASSERT_EQ(x1.extent2(),3);
    ASSERT_EQ(x1.extent3(),7);
  }
  {
    NumArray<int,ExtentsV<2,3,DynExtent>> x1;
    x1.resize(6);
    ASSERT_EQ(x1.extent0(),2);
    ASSERT_EQ(x1.extent1(),3);
    ASSERT_EQ(x1.extent2(),6);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray3,Misc)
{
  constexpr int nb_x = 3;
  constexpr int nb_y = 4;
  constexpr int nb_z = 5;

  NumArray<Int64,MDDim3> v(nb_x,nb_y,nb_z);
  v.fill(0);
  // Attention, v.extents() change si 'v' est redimensionné
  auto v_extents = v.extentsWithOffset();
  {
    for( Int32 x=0, xn=v.dim1Size(); x<xn; ++x ){
      for( Int32 y=0, yn=v.dim2Size(); y<yn; ++y ){
        for( Int32 z=0, zn=v.dim3Size(); z<zn; ++z ){
          ArrayIndex<3> idx{x,y,z};
          Int64 offset = v_extents.offset(idx);
          v(x,y,z) = offset;
          v({x,y,z}) = offset;
          v(idx) = offset;
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

namespace
{
NumArray<int,MDDim1> _createNumArray(Int32 size)
{
  std::cout << "IN_CREATE_1\n";
  NumArray<int,MDDim1> a(size);
  std::cout << "IN_CREATE_2\n";
  for(Int32 i=0; i<size; ++i )
    a(i) = size+i+2;
  return a;
}
}
TEST(NumArray3, Copy)
{
  int nb_x = 3;
  int nb_y = 4;
  int nb_z = 5;
  NumArray<Real, MDDim3> v(nb_x, nb_y, nb_z);
  v.fill(3.2);
  NumArray<Real, MDDim3> v2(nb_x * 2, nb_y / 2, nb_z * 3);

  v.copy(v2.span());

  {
    NumArray<int, MDDim1> vi0(4, { 1, 3, 5, 7 });
    NumArray<int, MDDim1> vi1(vi0);
    NumArray<int, MDDim1> vi2;
    vi2 = vi1;
    ASSERT_EQ(vi1.to1DSpan(), vi0.to1DSpan());
    ASSERT_EQ(vi2.to1DSpan(), vi1.to1DSpan());
    NumArray<int, MDDim1> vi3(vi0.to1DSpan());
    ASSERT_EQ(vi3.to1DSpan(), vi0.to1DSpan());

    Span<const int> vi0_span(vi0.to1DSmallSpan());
    Span<const int> vi1_span(vi1.to1DSmallSpan());
    ASSERT_EQ(vi0.to1DSpan(), vi0_span);
    ASSERT_EQ(vi1_span, vi1.to1DSpan());
    ASSERT_EQ(vi1.to1DSmallSpan(), vi0.to1DSmallSpan());
    ASSERT_EQ(vi1.to1DConstSmallSpan(), vi0.to1DConstSmallSpan());
    const NumArray<int, MDDim1>& v1_ref = vi1;
    Span<const int> vi1_ref_span(v1_ref.to1DSmallSpan());
    ASSERT_EQ(vi1_ref_span, vi1.to1DSpan());
  }

  {
    NumArray<int, MDDim1> vi0(4, { 1, 3, 5, 7 });
    NumArray<int, MDDim1> vi1(vi0);
    NumArray<int, MDDim1> vi2;
    vi2 = vi1;
    ASSERT_EQ(vi1.to1DSpan(), vi0.to1DSpan());
    ASSERT_EQ(vi2.to1DSpan(), vi1.to1DSpan());
  }
}

TEST(NumArray3,Move)
{
  // Test NumArray::NumArray(NumArray&&)
  {
    std::cout << "PART_1\n";
    NumArray<Int32,MDDim1> test_move(5);
    test_move.fill(3);
    Int32 wanted_size1 = 23;
    test_move = _createNumArray(wanted_size1);
    std::cout << "PART_2\n";
    ASSERT_EQ(test_move.totalNbElement(),wanted_size1) << "Bad size (test move 1)";
    ASSERT_EQ(test_move[6],wanted_size1+8) << "Bad size (test move 2)";
    Int32 wanted_size2 = 17;
    test_move = _createNumArray(wanted_size2);
    std::cout << "PART_3\n";
    ASSERT_EQ(test_move.totalNbElement(),wanted_size2) << "Bad size (test move 3)";
    ASSERT_EQ(test_move[3],wanted_size2+5) << "Bad size (test move 4)";
  }
  // Test NumArray::operator=(NumArray&&)
  {
    Int32 wanted_size1 = 31;
    std::cout << "PART_4\n";
    NumArray<Int32,MDDim1> test_move(_createNumArray(wanted_size1));
    std::cout << "PART_5\n";
    ASSERT_EQ(test_move.totalNbElement(),wanted_size1) << "Bad size (test move 1)";
    ASSERT_EQ(test_move[7],wanted_size1+9) << "Bad size (test move 2)";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray3,Index)
{
  ArrayIndex<3> index(1,4,2);
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
      a(i,j) = (i*253) + j;
    }
  }
}
template<typename T>
void _setNumArray3Values(T& a)
{
  for( Int32 i=0; i<a.dim1Size(); ++i ){
    for( Int32 j=0; j<a.dim2Size(); ++j ){
      for( Int32 k=0; k<a.dim3Size(); ++k ){
        a(i,j,k) = (i*253) + (j*27) + k;
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
    NumArray<Real,MDDim2,RightLayout> a(3,5);
    ASSERT_EQ(a.totalNbElement(),(3*5));
    _setNumArray2Values(a);
    auto values = a.to1DSpan();
    std::cout << "V=" << values << "\n";
    UniqueArray<Real> ref_value = { 0, 1, 2, 3, 4, 253, 254, 255, 256, 257, 506, 507, 508, 509, 510 };
    ASSERT_EQ(values.smallView(),ref_value.view());
  }

  {
    NumArray<Real,MDDim2,LeftLayout2> a(3,5);
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

template<typename NumArray3>
void _checkRightLayoutDim3(NumArray3& a)
{
  // Le tableau doit avoir les dimensions (2,3,5);
  ASSERT_EQ(a.totalNbElement(),(2*3*5));
  ASSERT_EQ(a.extent0(),2);
  ASSERT_EQ(a.extent1(),3);
  ASSERT_EQ(a.extent2(),5);
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

template<typename NumArray3>
void _checkLeftLayoutDim3(NumArray3& a)
{
  // Le tableau doit avoir les dimensions (2,3,5);
  //NumArray<Real,MDDim3,LeftLayout3> a(2,3,5);
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

TEST(NumArray3,Layout)
{
  std::cout << "TEST_NUMARRAY3 Layout\n";

  {
    NumArray<Real,MDDim3,RightLayout3> a(2,3,5);
    std::cout << "TEST_NUMARRAY3 RightLayout 1\n";
    _checkRightLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<DynExtent,3,5>,RightLayout3> a(2);
    std::cout << "TEST_NUMARRAY3 RightLayout 2\n";
    _checkRightLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<2,DynExtent,5>,RightLayout3> a(3);
    std::cout << "TEST_NUMARRAY3 RightLayout 3\n";
    _checkRightLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<2,3,5>,RightLayout3> a;
    std::cout << "TEST_NUMARRAY3 RightLayout 4\n";
    _checkRightLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<DynExtent,3,DynExtent>,RightLayout3> a(2,5);
    std::cout << "TEST_NUMARRAY3 RightLayout 5\n";
    _checkRightLayoutDim3(a);
  }

  {
    NumArray<Real,MDDim3,LeftLayout3> a(2,3,5);
    std::cout << "TEST_NUMARRAY3 LeftLayout 1\n";
    _checkLeftLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<DynExtent,3,5>,LeftLayout3> a(2);
    std::cout << "TEST_NUMARRAY3 LeftLayout 2\n";
    _checkLeftLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<2,DynExtent,5>,LeftLayout3> a(3);
    std::cout << "TEST_NUMARRAY3 LeftLayout 3\n";
    _checkLeftLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<2,3,5>,LeftLayout3> a;
    std::cout << "TEST_NUMARRAY3 LeftLayout 4\n";
    _checkLeftLayoutDim3(a);
  }
  {
    NumArray<Real,ExtentsV<DynExtent,3,DynExtent>,LeftLayout3> a(2,5);
    std::cout << "TEST_NUMARRAY3 LeftLayout 5\n";
    _checkLeftLayoutDim3(a);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray,RealN)
{
  {
    NumArray<Real2,MDDim1> a(5);
    a(2) = Real2(0.0,3.2);
    a(3)(1) = 2.0;
    ASSERT_EQ(a(3).y,2.0);
  }

  {
    NumArray<Real3,MDDim1> a(5);
    const Real3 v(0.0,3.2,5.6);
    a(0) = v;
    a(4)(1) = 4.0;
    ASSERT_EQ(a(4).y,4.0);
    ASSERT_EQ(a(0),v);
  }

  {
    NumArray<Real2x2,MDDim1> a(5);
    const Real2 v0(1.2,1.7);
    const Real2x2 v(Real2(3.2,5.6), Real2(3.4,1.7));
    a(0) = v;
    a(3)(1,0) = v0.x;
    a(3)(1,1) = v0.y;
    a(4)(1,0) = 4.0;
    ASSERT_EQ(a(4).y.x,4.0);
    ASSERT_EQ(a(0),v);
    ASSERT_EQ(a(3)(1),v0);
  }

  {
    NumArray<Real3x3,MDDim1> a(5);
    const Real3 v0(1.2,3.4,1.7);
    const Real3x3 v(Real3(0.0,3.2,5.6), Real3(1.2,3.4,1.7), Real3(9.2,1.4,5.0));
    a(0) = v;
    a(3)(1)(0) = v0.x;
    a(3)(1)(1) = v0.y;
    a(3)(1)(2) = v0.z;
    a(4)(1)(2) = 4.0;
    ASSERT_EQ(a(4).y.z,4.0);
    ASSERT_EQ(a(0),v);
    ASSERT_EQ(a(3)(1),v0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray, ReadFromText)
{
  {
    const char* values1_str = "1 3 -2 \n -7 -5 12 \n 3 9 11\n";
    NumArray<Int32, MDDim1> ref_value(9, { 1, 3, -2, -7, -5, 12, 3, 9, 11 });
    std::istringstream istr1(values1_str);
    NumArray<Int32, MDDim1> int32_values;
    NumArrayUtils::readFromText(int32_values, istr1);
    ASSERT_EQ(int32_values.extent0(), 9);
    ASSERT_EQ(int32_values.to1DSpan(), ref_value.to1DSpan());
  }
  {
    const char* values1_str = "1.1 3.3 -2.5 \n \n 2.1 4.99 12.23 \n 23 \n 45.1 11.9e2 -12.6e4\n";
    NumArray<Real, MDDim1> ref_value(10, { 1.1, 3.3, -2.5, 2.1, 4.99, 12.23, 23, 45.1, 11.9e2, -12.6e4 });
    std::istringstream istr1(values1_str);
    NumArray<Real, MDDim1> real_values;
    NumArrayUtils::readFromText(real_values, istr1);
    ASSERT_EQ(real_values.extent0(), 10);
    ASSERT_EQ(real_values.to1DSpan(), ref_value.to1DSpan());
  }
}
namespace TestCopyNumArray
{
using Real = double;

struct B
{
  Arcane::NumArray<Real, Arcane::MDDim1> a_;
};

auto bar()
{
  auto tpq = Arcane::NumArray<Real, Arcane::MDDim1>(5);
  tpq.fill(1.526);
  auto const& w = tpq;
  std::cout << w.to1DSpan() << "\n";
  B b{ w };
  std::cout << b.a_.to1DSpan() << "\n";
  return b;
}

auto bar2()
{
  auto tpq = Arcane::NumArray<Real, Arcane::MDDim1>(5);
  tpq.fill(1.526);
  std::cout << tpq.to1DSpan() << "\n";
  return tpq;
}

} // namespace TestCopyNumArray

TEST(NumArray, TestCopy)
{
  using namespace TestCopyNumArray;

  auto test4 = bar();
  std::cout << "Val 4 = "
            << &test4.a_
            << " " << test4.a_.to1DSpan() << "\n";
  auto test5 = bar2();
  std::cout << "Val 5 = " << &test5 << " " << test5.to1DSpan() << "\n";
  ASSERT_EQ(test4.a_.to1DSpan(), test5.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// On instantie explicitement pour tester que toutes les méthodes templates sont valides
template class NumArray<float,MDDim4,RightLayout>;
template class NumArray<float,MDDim3,RightLayout>;
template class NumArray<float,MDDim2,RightLayout>;

template class NumArray<float,MDDim4,LeftLayout>;
template class NumArray<float,MDDim3,LeftLayout>;
template class NumArray<float,MDDim2,LeftLayout>;

template class NumArray<float,MDDim1>;

template class NumArray<float,ExtentsV<7,DynExtent,2,3>,RightLayout>;
template class NumArray<float,ExtentsV<DynExtent,2,3>,RightLayout>;
template class NumArray<float,ExtentsV<2,3>>;
template class NumArray<float,ExtentsV<2>>;
template class NumArray<float,ExtentsV<3,DynExtent>>;

template class MDSpan<float,MDDim4,RightLayout>;
template class MDSpan<float,MDDim3,RightLayout>;
template class MDSpan<float,MDDim2,RightLayout>;

template class MDSpan<float,MDDim4,LeftLayout>;
template class MDSpan<float,MDDim3,LeftLayout>;
template class MDSpan<float,MDDim2,LeftLayout>;

template class MDSpan<float,MDDim1>;

template class MDSpan<float,ExtentsV<7,DynExtent,2,3>,RightLayout>;
template class MDSpan<float,ExtentsV<DynExtent,2,3>,RightLayout>;
template class MDSpan<float,ExtentsV<2,3>>;
template class MDSpan<float,ExtentsV<2>>;
template class MDSpan<float,ExtentsV<3,DynExtent>>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
