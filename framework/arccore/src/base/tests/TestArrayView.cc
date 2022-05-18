// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/Span.h"
#include "arccore/base/ArrayView.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array3View,Misc)
{
  using namespace Arccore;
  int nb_x = 3;
  int nb_y = 4;
  int nb_z = 5;
  std::vector<Int32> buf(nb_x*nb_y*nb_z);
  for( size_t i=0, n=buf.size(); i<n; ++i )
    buf[i] = (Int32)(i+1);

  ConstArray3View<Int32> v(buf.data(),nb_x,nb_y,nb_z);
  Integer global_index = 0;
  for( Integer x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Integer y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Integer z=0, zn=v.dim3Size(); z<zn; ++z ){
        ++global_index;
        Int32 val1 = v[x][y][z];
        Int32 val2 = v.item(x,y,z);
        std::cout  << " V=" << val1 << " x=" << x << " y=" << y << " z=" << z << '\n';
        ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
        ASSERT_TRUE(val1==global_index) << "Bad value v1=" << val1 << " expected=" << global_index;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array4View,Misc)
{
  using namespace Arccore;
  int nb_x = 2;
  int nb_y = 3;
  int nb_z = 4;
  int nb_a = 5;
  std::vector<Int32> buf(nb_x*nb_y*nb_z*nb_a);
  for( size_t i=0, n=buf.size(); i<n; ++i )
    buf[i] = (Int32)(i+1);

  ConstArray4View<Int32> v(buf.data(),nb_x,nb_y,nb_z,nb_a);
  Integer global_index = 0;
  for( Integer x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Integer y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Integer z=0, zn=v.dim3Size(); z<zn; ++z ){
        for( Integer a=0, an=v.dim4Size(); a<an; ++a ){
          ++global_index;
          Int32 val1 = v[x][y][z][a];
          Int32 val2 = v.item(x,y,z,a);
          std::cout << " V=" << val1 << " x=" << x << " y=" << y << " z=" << z << " a=" << a << '\n';
          ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
          ASSERT_TRUE(val1==global_index) << "Bad value v1=" << val1 << " expected=" << global_index;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<typename T> void
_testIterator(T values)
{
  {
    auto r1 = std::make_reverse_iterator(values.end());
    auto r2 = std::make_reverse_iterator(values.begin());
    for( ; r1!=r2; ++ r1 ){
      std::cout << "RVALUE = " << *r1 << '\n';
    }
  }
  {
    auto r1 = values.rbegin();
    ASSERT_EQ((*r1),7);
    ++r1;
    ASSERT_EQ((*r1),9);
    ++r1;
    ASSERT_EQ((*r1),4);
    ++r1;
    ASSERT_TRUE((r1==values.rend()));
  }
}

}

TEST(ArrayView,Iterator)
{
  using namespace Arccore;

  std::vector<Arccore::Int32> vector_values = { 4, 9, 7 };
  Integer vec_size = arccoreCheckArraySize(vector_values.size());

  ArrayView<Int32> values1(vec_size,vector_values.data());
  _testIterator(values1);

  ConstArrayView<Int32> values2(vec_size,vector_values.data());
  _testIterator(values2);

  Span<Int32> values3(vector_values.data(),vector_values.size());
  _testIterator(values3);

  Span<const Int32> values4(vector_values.data(),vector_values.size());
  _testIterator(values4);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Span,Convert)
{
  using namespace Arccore;
  std::vector<Int64> vector_values = { 5, 7, 11 };
  Int32 vector_size = static_cast<Int32>(vector_values.size());
  ArrayView<Int64> a_view(vector_size,vector_values.data());
  ASSERT_EQ(a_view.size(),vector_size) << "Bad a_view size";
  ASSERT_EQ(a_view[0],vector_values[0]) << "Bad a_view[0]";
  ASSERT_EQ(a_view[1],vector_values[1]) << "Bad a_view[1]";
  ASSERT_EQ(a_view[2],vector_values[2]) << "Bad a_view[2]";

  ConstArrayView<Int64> a_const_view(vector_size,vector_values.data());
  ASSERT_EQ(a_const_view.size(),vector_size) << "Bad a_const_view size";
  ASSERT_EQ(a_const_view[0],vector_values[0]) << "Bad a_const_view[0]";
  ASSERT_EQ(a_const_view[1],vector_values[1]) << "Bad a_const_view[1]";
  ASSERT_EQ(a_const_view[2],vector_values[2]) << "Bad a_const_view[2]";

  Span<const Int64> a_const_span(vector_values.data(),vector_values.size());
  Span<Int64> a_span(vector_values.data(),vector_values.size());
  ByteConstSpan a_const_bytes = asBytes(a_const_span);
  ByteSpan a_bytes = asWritableBytes(a_span);
  ASSERT_EQ(a_const_bytes.size(),24) << "Bad a_const_bytes_size (1)";
  ASSERT_EQ(a_bytes.size(),24) << "Bad a_bytes_size (2)";
  Span<Int64> span2(a_view);
  Span<const Int64> span3(a_view);
  Span<const Int64> span4(a_const_view);
  ASSERT_EQ(span2.size(),a_view.size()) << "Bad span2 size";
  ASSERT_EQ(span3.size(),a_view.size()) << "Bad span3 size";
  ASSERT_EQ(span4.size(),a_const_view.size()) << "Bad span4 size";
  span3 = a_const_view;
  span3 = a_view;
  span2 = a_view;
  ASSERT_EQ(span2.size(),a_view.size()) << "Bad span2 (2) size";
  ASSERT_EQ(span3.size(),a_view.size()) << "Bad span3 (2) size";
  std::cout << "View=" << a_view << '\n';
  std::cout << "ConstView=" << a_const_view << '\n';
  std::cout << "Span3=" << span3 << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Vérifie que \a a1 et \a a2 sont identiques
template<typename A1,typename A2>
void _checkSame(A1& a1,A2& a2,const char* message)
{
  using namespace Arccore;
  using size_type = typename A1::size_type;
  Int64 s1 = a1.size();
  Int64 s2 = a2.size();
  ASSERT_EQ(s1,s2) << "Bad size " << message;
  for( size_type i=0, n=a1.size(); i<n; ++i )
    ASSERT_EQ(a1[i],a2[i]) << "Bad value[" << i << "]" << message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(ArrayView,StdArray)
{
  using namespace Arccore;
  std::array<Int64,0> v0;
  std::array<Int64,2> v1 { 5, 7 };
  std::array<Int64,3> v2 { 2, 4, -2 };
  std::array<const Int64,4> v3 { 9, 13, 32, 27 };

  {
    ArrayView<Int64> view0 { v0 };
    _checkSame(view0,v0,"view0==v0");

    ArrayView<Int64> view1 { v1 };
    _checkSame(view1,v1,"view1==v1");

    view0 = v2;
    _checkSame(view0,v2,"view0==v2");
  }

  {
    ConstArrayView<Int64> view0 { v0 };
    _checkSame(view0,v0,"const view0==v0");

    ConstArrayView<Int64> view1 { v1 };
    _checkSame(view1,v1,"const view1==v1");

    view0 = v2;
    _checkSame(view0,v2,"const view0==v2");

    ConstArrayView<Int64> view2 { v3 };
    _checkSame(view2,v3,"const view2==v3");

    view1 = v3;
    _checkSame(view1,v3,"const view1==v3");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SpanType,typename ConstSpanType>
void
_testSpanStdArray()
{
  using namespace Arccore;
  std::array<Int64,0> v0;
  std::array<Int64,2> v1 { 5, 7 };
  std::array<Int64,3> v2 { 2, 4, -2 };
  std::array<const Int64,4> v3 { 9, 13, 32, 27 };

  {
    SpanType span0 { v0 };
    _checkSame(span0,v0,"span0==v0");

    SpanType span1 { v1 };
    _checkSame(span1,v1,"span1==v1");

    SpanType span2 { v1 };
    ASSERT_TRUE(span1==span2);
    ASSERT_FALSE(span1!=span2);

    SpanType const_span2 { v1 };
    ASSERT_TRUE(span1==const_span2);
    ASSERT_FALSE(span1!=const_span2);

    span0 = v2;
    _checkSame(span0,v2,"span0==v2");
  }

  {
    ConstSpanType span0 { v0 };
    _checkSame(span0,v0,"const span0==v0");

    ConstSpanType span1 { v1 };
    _checkSame(span1,v1,"const span1==v1");

    span0 = v2;
    _checkSame(span0,v2,"const span0==v2");

    ConstSpanType span2 { v3 };
    _checkSame(span2,v3,"const span2==v3");

    ConstSpanType span3 { v3 };
    ASSERT_TRUE(span2==span3);
    ASSERT_FALSE(span2!=span3);

    span1 = v3;
    _checkSame(span1,v3,"const span1==v3");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Span,StdArray)
{
  using namespace Arccore;
  _testSpanStdArray<Span<Int64>,Span<const Int64>>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(SmallSpan,StdArray)
{
  using namespace Arccore;
  _testSpanStdArray<SmallSpan<Int64>,SmallSpan<const Int64>>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ViewType> void
_testSubViewInterval()
{
  using namespace Arccore;
  
  std::array<Int64,12> vlist { 9, 13, 32, 27, 43, -5, 2, -7, 8, 11, 25, 48 };
  ViewType view{vlist};

  {
    ViewType null_view;
    ViewType sub_view0{view.subViewInterval(1,0)};
    ViewType sub_view1{view.subViewInterval(-1,2)};
    ViewType sub_view2{view.subViewInterval(2,2)};
    ViewType sub_view3{view.subViewInterval(0,0)};
    std::cout << "SUBVIEW0=" << sub_view1 << '\n';
    _checkSame(sub_view0,null_view,"sub_view0_null");
    _checkSame(sub_view1,null_view,"sub_view1_null");
    _checkSame(sub_view2,null_view,"sub_view2_null");
    _checkSame(sub_view3,null_view,"sub_view3_null");
  }

  {
    std::array<Int64,4> vlist0 { 9, 13, 32, 27 };
    std::array<Int64,4> vlist1 { 43, -5, 2, -7 };
    std::array<Int64,4> vlist2 { 8, 11, 25, 48 };

    ViewType sub_view0{view.subViewInterval(0,3)};
    ViewType sub_view1{view.subViewInterval(1,3)};
    ViewType sub_view2{view.subViewInterval(2,3)};
    std::cout << "SUBVIEW1=" << sub_view1 << '\n';
    _checkSame(sub_view0,vlist0,"sub_view0");
    _checkSame(sub_view1,vlist1,"sub_view1");
    _checkSame(sub_view2,vlist2,"sub_view2");
  }

  {
    std::array<Int64,2> vlist0 { 9, 13 };
    std::array<Int64,2> vlist1 { 32, 27 };
    std::array<Int64,2> vlist2 { 43, -5 };
    std::array<Int64,2> vlist3 { 2, -7 };
    std::array<Int64,4> vlist4 { 8, 11, 25, 48 };

    ViewType sub_view0{view.subViewInterval(0,5)};
    ViewType sub_view1{view.subViewInterval(1,5)};
    ViewType sub_view2{view.subViewInterval(2,5)};
    ViewType sub_view3{view.subViewInterval(3,5)};
    ViewType sub_view4{view.subViewInterval(4,5)};
    std::cout << "SUBVIEW2=" << sub_view1 << '\n';
    _checkSame(sub_view0,vlist0,"sub_view0");
    _checkSame(sub_view1,vlist1,"sub_view1");
    _checkSame(sub_view2,vlist2,"sub_view2");
    _checkSame(sub_view3,vlist3,"sub_view3");
    _checkSame(sub_view4,vlist4,"sub_view4");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(ArrayView,SubViewInterval)
{
  using namespace Arccore;
  _testSubViewInterval<ArrayView<Int64>>();
  _testSubViewInterval<ConstArrayView<Int64>>();
  _testSubViewInterval<Span<Int64>>();
  _testSubViewInterval<SmallSpan<Int64>>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
template class ArrayView<Int32>;
template class ConstArrayView<Int32>;
template class ArrayView<double>;
template class ConstArrayView<double>;

template class Span<Int32>;
template class Span<const Int32>;
template class Span<double>;
template class Span<const double>;

template class SmallSpan<Int32>;
template class SmallSpan<const Int32>;
template class SmallSpan<double>;
template class SmallSpan<const double>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
