// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/Span.h"
#include "arccore/base/Span2.h"
#include "arccore/base/ArrayView.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"

#include <vector>
#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array2View,Misc)
{
  using namespace Arccore;
  int nb_x = 3;
  int nb_y = 4;
  std::vector<Int32> buf(nb_x*nb_y);
  for( size_t i=0, n=buf.size(); i<n; ++i )
    buf[i] = (Int32)(i+1);

  ConstArray2View<Int32> v(buf.data(),nb_x,nb_y);
  Integer global_index = 0;
  for( Integer x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Integer y=0, yn=v.dim2Size(); y<yn; ++y ){
      ++global_index;
      Int32 val1 = v[x][y];
      Int32 val2 = v.item(x,y);
      std::cout  << " V=" << val1 << " x=" << x << " y=" << y;
      ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
      ASSERT_TRUE(val1==global_index) << "Bad value v1=" << val1 << " expected=" << global_index;
      ASSERT_EQ(v(x,y),val1);
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
      bool is_ok = v[x,y]==val1;
      ASSERT_TRUE(is_ok);
#endif
    }
  }
}

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
        ASSERT_EQ(v(x,y,z),val1);
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
        bool is_ok = v[x,y,z] == val1;
        ASSERT_TRUE(is_ok);
#endif
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
          ASSERT_EQ(v(x,y,z,a),val1);
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
          bool is_ok = v[x,y,z,a] == val1;
          ASSERT_TRUE(is_ok);
#endif
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
  ByteConstSpan a_const_bytes2 = asBytes(a_span);
  //ByteSpan a_bytes2 = asWritableBytes(a_const_span);
  ASSERT_EQ(a_const_bytes.size(),24) << "Bad a_const_bytes_size (1)";
  ASSERT_EQ(a_const_bytes2.size(),24) << "Bad a_const_bytes2_size (1)";
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

// Vérifie que \a a1 et \a a2 sont identiques
template<typename A1,typename A2>
void _checkSame2(A1& a1,A2& a2,const char* message)
{
  using namespace Arccore;
  using size_type = typename A1::size_type;
  const Int64 s1_dim1 = a1.dim1Size();
  const Int64 s2_dim1 = a2.dim1Size();
  ASSERT_EQ(s1_dim1,s2_dim1) << "Bad size " << message;
  const Int64 s1_dim2 = a1.dim2Size();
  const Int64 s2_dim2 = a2.dim2Size();
  ASSERT_EQ(s1_dim2,s2_dim2) << "Bad size " << message;
  for( size_type i=0; i<s1_dim1; ++i )
    for( size_type j=0; j<s1_dim2; ++j )
      ASSERT_EQ(a1[i][j],a2[i][j]) << "Bad value[" << i << ',' << j << "]" << message;
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

template<typename SpanType,typename ConstSpanType> void
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
  {
    SpanType span1 { v1 };
    ConstSpanType const_span1 { v1 };
    ASSERT_TRUE(span1==const_span1);

    SpanType span2 { v2 };
    ConstSpanType const_span3 { v3 };
    ASSERT_TRUE(span2!=const_span3);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Span, StdArray)
{
  using namespace Arccore;
  _testSpanStdArray<Span<Int64>, Span<const Int64>>();

  std::array<Int64, 0> v0;
  std::array<Int64, 2> v1{ 5, 7 };
  std::array<Int64, 3> v2{ 2, 4, -2 };
  auto s0 = asSpan(v0);
  _checkSame(s0, v0, "s0==v0");
  auto s1 = asSpan(v1);
  _checkSame(s1, v1, "s1==v1");
  auto s2 = asSpan(v2);
  _checkSame(s2, v2, "s2==v2");

  auto sp0 = asSmallSpan(v0);
  _checkSame(sp0, v0, "s0==v0");
  auto sp1 = asSmallSpan(v1);
  _checkSame(sp1, v1, "s1==v1");
  auto sp2 = asSmallSpan(v2);
  _checkSame(sp2, v2, "s2==v2");

  {
    Span<Int64> ss0(s0);
    auto bytes = asBytes(ss0);
    auto x = asSpan<Int64>(bytes);
    ASSERT_EQ(x.data(), nullptr);
    ASSERT_EQ(x.size(), 0);
  }
  {
    SmallSpan<Int64> ssp0(sp0);
    auto bytes = asBytes(ssp0);
    auto x = asSmallSpan<Int64>(bytes);
    ASSERT_EQ(x.data(), nullptr);
    ASSERT_EQ(x.size(), 0);
  }

  {
    auto bytes = asWritableBytes(s1);
    auto x = asSpan<Int64>(bytes);
    _checkSame(x, v1, "x==v1");
  }

  {
    auto bytes = asWritableBytes(s2);
    auto x = asSpan<Int64>(bytes);
    _checkSame(x, v2, "x==v2");
    x[2] = 5;
    ASSERT_EQ(v2[2], 5);
  }

  {
    auto bytes = asWritableBytes(sp2);
    auto x = asSmallSpan<Int64>(bytes);
    _checkSame(x, v2, "x==v2");
    x[2] = 5;
    ASSERT_EQ(v2[2], 5);
  }

  {
    std::array<Int64,2> v1 { 5, 7 };
    std::array<Int64,3> v2 { 2, 4, -2 };
    Span<Int64,2> fixed_s1(v1);
    Span<Int64,3> fixed_s2(v2);
    ASSERT_FALSE(fixed_s1==fixed_s2);
    ASSERT_TRUE(fixed_s1!=fixed_s2);

    LargeSpan<const Int64> s1_a(s1);
    Span<const std::byte> fb1(asBytes(s1_a));
    Span<std::byte> fb2(asWritableBytes(s2));
    ASSERT_FALSE(fb1==fb2);

    std::array<Real, 3> v2r{ 2.0, 4.1, -2.3 };
    SmallSpan<const Real> small2(v2r);
    LargeSpan<const std::byte> small_fb2(asBytes(small2));

    Span<Int64,DynExtent,-1> a4(v2.data()+1,v2.size()-1);
    ASSERT_EQ(a4[-1],2);
  }
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

template<typename SpanType,typename ConstSpanType> void
_testSpan2StdArray()
{
  using namespace Arccore;

  std::array<Int64,6> v1 { 5, 7, 9, 32, -5, -6 };
  std::array<Int64,5> v2 { 1, 9, 32, 41, -5 };
  std::array<const Int64,12> v3 { 12, 33, 47, 55, 36, 13, 9, 7, 5, 1, 45, 38 };

  SpanType s0 { };
  SpanType s1 { v1.data(), 3, 2 };
  SpanType s2 { v2.data(), 1, 5 };
  ConstSpanType s3 { v3.data(), 4, 3 };

  ASSERT_EQ(s1[2][1],s1(2,1));

#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  {
    bool is_ok = s1[2,1]==s1(2,1);
    ASSERT_TRUE(is_ok);
  }
#endif

  {
    SpanType span0 { s0 };
    _checkSame2(span0,s0,"span0==s0");

    SpanType span1 { s1 };
    _checkSame2(span1,s1,"span1==s1");

    SpanType span2 { s1 };
    ASSERT_TRUE(span1==span2);
    ASSERT_FALSE(span1!=span2);

    SpanType const_span2 { s1 };
    ASSERT_TRUE(span1==const_span2);
    ASSERT_FALSE(span1!=const_span2);
  }

  {
    ConstSpanType span0 { s0 };
    _checkSame2(span0,s0,"const span0==s0");

    ConstSpanType span1 { s1 };
    _checkSame2(span1,s1,"const span1==s1");

    span0 = s2;
    _checkSame2(span0,s2,"const span0==s2");

    ConstSpanType span2 { s3 };
    _checkSame2(span2,s3,"const span2==s3");

    ConstSpanType span3 { s3 };
    ASSERT_TRUE(span2==span3);
    ASSERT_FALSE(span2!=span3);

    span1 = s3;
    _checkSame2(span1,s3,"const span1==s3");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Span2,StdArray)
{
  using namespace Arccore;
  _testSpan2StdArray<Span2<Int64>,Span2<const Int64>>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(SmallSpan2,StdArray)
{
  using namespace Arccore;
  _testSpan2StdArray<SmallSpan2<Int64>,SmallSpan2<const Int64>>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ViewType> void
_testSubViewInterval()
{
  using namespace Arccore;

  std::array<Int64, 12> vlist{ 9, 13, 32, 27, 43, -5, 2, -7, 8, 11, 25, 48 };
  ViewType view{ vlist };

  {
    ViewType null_view;
    ViewType sub_view0{ view.subViewInterval(1, 0) };
    ViewType sub_view1{ view.subViewInterval(-1, 2) };
    ViewType sub_view2{ view.subViewInterval(2, 2) };
    ViewType sub_view3{ view.subViewInterval(0, 0) };
    std::cout << "SUBVIEW0=" << sub_view1 << '\n';
    _checkSame(sub_view0, null_view, "sub_view0_null");
    _checkSame(sub_view1, null_view, "sub_view1_null");
    _checkSame(sub_view2, null_view, "sub_view2_null");
    _checkSame(sub_view3, null_view, "sub_view3_null");
  }

  {
    std::array<Int64, 4> vlist0{ 9, 13, 32, 27 };
    std::array<Int64, 4> vlist1{ 43, -5, 2, -7 };
    std::array<Int64, 4> vlist2{ 8, 11, 25, 48 };

    ViewType sub_view0{ view.subViewInterval(0, 3) };
    ViewType sub_view1{ view.subViewInterval(1, 3) };
    ViewType sub_view2{ view.subViewInterval(2, 3) };
    std::cout << "SUBVIEW1=" << sub_view1 << '\n';
    _checkSame(sub_view0, vlist0, "sub_view0");
    _checkSame(sub_view1, vlist1, "sub_view1");
    _checkSame(sub_view2, vlist2, "sub_view2");
  }

  {
    std::array<Int64, 2> vlist0{ 9, 13 };
    std::array<Int64, 2> vlist1{ 32, 27 };
    std::array<Int64, 2> vlist2{ 43, -5 };
    std::array<Int64, 2> vlist3{ 2, -7 };
    std::array<Int64, 4> vlist4{ 8, 11, 25, 48 };

    ViewType sub_view0{ view.subViewInterval(0, 5) };
    ViewType sub_view1{ view.subViewInterval(1, 5) };
    ViewType sub_view2{ view.subViewInterval(2, 5) };
    ViewType sub_view3{ view.subViewInterval(3, 5) };
    ViewType sub_view4{ view.subViewInterval(4, 5) };
    std::cout << "SUBVIEW2=" << sub_view1 << '\n';
    _checkSame(sub_view0, vlist0, "sub_view0");
    _checkSame(sub_view1, vlist1, "sub_view1");
    _checkSame(sub_view2, vlist2, "sub_view2");
    _checkSame(sub_view3, vlist3, "sub_view3");
    _checkSame(sub_view4, vlist4, "sub_view4");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ViewType> void
_testSubSpanInterval()
{
  using namespace Arccore;

  std::array<Int64, 12> vlist{ 9, 13, 32, 27, 43, -5, 2, -7, 8, 11, 25, 48 };
  ViewType view{ vlist };

  {
    ViewType null_view;
    ViewType sub_view0{ view.subSpanInterval(1, 0) };
    ViewType sub_view1{ view.subSpanInterval(-1, 2) };
    ViewType sub_view2{ view.subSpanInterval(2, 2) };
    ViewType sub_view3{ view.subSpanInterval(0, 0) };
    std::cout << "SUBVIEW0=" << sub_view1 << '\n';
    _checkSame(sub_view0, null_view, "sub_view0_null");
    _checkSame(sub_view1, null_view, "sub_view1_null");
    _checkSame(sub_view2, null_view, "sub_view2_null");
    _checkSame(sub_view3, null_view, "sub_view3_null");
  }

  {
    std::array<Int64, 4> vlist0{ 9, 13, 32, 27 };
    std::array<Int64, 4> vlist1{ 43, -5, 2, -7 };
    std::array<Int64, 4> vlist2{ 8, 11, 25, 48 };

    ViewType sub_view0{ view.subSpanInterval(0, 3) };
    ViewType sub_view1{ view.subSpanInterval(1, 3) };
    ViewType sub_view2{ view.subSpanInterval(2, 3) };
    std::cout << "SUBVIEW1=" << sub_view1 << '\n';
    _checkSame(sub_view0, vlist0, "sub_view0");
    _checkSame(sub_view1, vlist1, "sub_view1");
    _checkSame(sub_view2, vlist2, "sub_view2");
  }

  {
    std::array<Int64, 2> vlist0{ 9, 13 };
    std::array<Int64, 2> vlist1{ 32, 27 };
    std::array<Int64, 2> vlist2{ 43, -5 };
    std::array<Int64, 2> vlist3{ 2, -7 };
    std::array<Int64, 4> vlist4{ 8, 11, 25, 48 };

    ViewType sub_view0{ view.subSpanInterval(0, 5) };
    ViewType sub_view1{ view.subSpanInterval(1, 5) };
    ViewType sub_view2{ view.subSpanInterval(2, 5) };
    ViewType sub_view3{ view.subSpanInterval(3, 5) };
    ViewType sub_view4{ view.subSpanInterval(4, 5) };
    std::cout << "SUBVIEW2=" << sub_view1 << '\n';
    _checkSame(sub_view0, vlist0, "sub_view0");
    _checkSame(sub_view1, vlist1, "sub_view1");
    _checkSame(sub_view2, vlist2, "sub_view2");
    _checkSame(sub_view3, vlist3, "sub_view3");
    _checkSame(sub_view4, vlist4, "sub_view4");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ViewType> void
_testSubPartInterval()
{
  using namespace Arccore;

  std::array<Int64, 12> vlist{ 9, 13, 32, 27, 43, -5, 2, -7, 8, 11, 25, 48 };
  ViewType view{ vlist };

  {
    ViewType null_view;
    ViewType sub_view0{ view.subPartInterval(1, 0) };
    ViewType sub_view1{ view.subPartInterval(-1, 2) };
    ViewType sub_view2{ view.subPartInterval(2, 2) };
    ViewType sub_view3{ view.subPartInterval(0, 0) };
    std::cout << "SUBVIEW0=" << sub_view1 << '\n';
    _checkSame(sub_view0, null_view, "sub_view0_null");
    _checkSame(sub_view1, null_view, "sub_view1_null");
    _checkSame(sub_view2, null_view, "sub_view2_null");
    _checkSame(sub_view3, null_view, "sub_view3_null");
  }

  {
    std::array<Int64, 4> vlist0{ 9, 13, 32, 27 };
    std::array<Int64, 4> vlist1{ 43, -5, 2, -7 };
    std::array<Int64, 4> vlist2{ 8, 11, 25, 48 };

    ViewType sub_view0{ view.subPartInterval(0, 3) };
    ViewType sub_view1{ view.subPartInterval(1, 3) };
    ViewType sub_view2{ view.subPartInterval(2, 3) };
    std::cout << "SUBVIEW1=" << sub_view1 << '\n';
    _checkSame(sub_view0, vlist0, "sub_view0");
    _checkSame(sub_view1, vlist1, "sub_view1");
    _checkSame(sub_view2, vlist2, "sub_view2");
  }

  {
    std::array<Int64, 2> vlist0{ 9, 13 };
    std::array<Int64, 2> vlist1{ 32, 27 };
    std::array<Int64, 2> vlist2{ 43, -5 };
    std::array<Int64, 2> vlist3{ 2, -7 };
    std::array<Int64, 4> vlist4{ 8, 11, 25, 48 };

    ViewType sub_view0{ view.subPartInterval(0, 5) };
    ViewType sub_view1{ view.subPartInterval(1, 5) };
    ViewType sub_view2{ view.subPartInterval(2, 5) };
    ViewType sub_view3{ view.subPartInterval(3, 5) };
    ViewType sub_view4{ view.subPartInterval(4, 5) };
    std::cout << "SUBVIEW2=" << sub_view1 << '\n';
    _checkSame(sub_view0, vlist0, "sub_view0");
    _checkSame(sub_view1, vlist1, "sub_view1");
    _checkSame(sub_view2, vlist2, "sub_view2");
    _checkSame(sub_view3, vlist3, "sub_view3");
    _checkSame(sub_view4, vlist4, "sub_view4");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(ArrayView, SubViewInterval)
{
  using namespace Arccore;
  _testSubViewInterval<ArrayView<Int64>>();
  _testSubViewInterval<ConstArrayView<Int64>>();
  _testSubSpanInterval<Span<Int64>>();
  _testSubSpanInterval<SmallSpan<Int64>>();

  _testSubPartInterval<ArrayView<Int64>>();
  _testSubPartInterval<ConstArrayView<Int64>>();
  _testSubPartInterval<Span<Int64>>();
  _testSubPartInterval<SmallSpan<Int64>>();
}

TEST(ArrayView,Copyable)
{
  using namespace Arccore;
  ASSERT_TRUE(std::is_trivially_copyable_v<ArrayView<int>>);
  ASSERT_TRUE(std::is_trivially_copyable_v<ConstArrayView<int>>);

  ASSERT_TRUE(std::is_trivially_copyable_v<Array2View<int>>);
  ASSERT_TRUE(std::is_trivially_copyable_v<ConstArray2View<int>>);

  ASSERT_TRUE(std::is_trivially_copyable_v<Array3View<int>>);
  ASSERT_TRUE(std::is_trivially_copyable_v<ConstArray3View<int>>);

  ASSERT_TRUE(std::is_trivially_copyable_v<Array4View<int>>);
  ASSERT_TRUE(std::is_trivially_copyable_v<ConstArray4View<int>>);

  ASSERT_TRUE(std::is_trivially_copyable_v<Span<int>>);
  ASSERT_TRUE(std::is_trivially_copyable_v<Span<const int>>);

  ASSERT_TRUE(std::is_trivially_copyable_v<Span2<int>>);
  ASSERT_TRUE(std::is_trivially_copyable_v<Span2<const int>>);
}

TEST(Span,FixedValue)
{
  using namespace Arccore;
  std::cout << "sizeof(Span<Int32,1>) = " << sizeof(Span<Int32,1>) << "\n";
  std::cout << "sizeof(Span<Int32,DynExtent>) = " << sizeof(Span<Int32,DynExtent>) << "\n";
  std::cout << "sizeof(Span<Int64,1>) = " << sizeof(Span<Int64,1>) << "\n";
  std::cout << "sizeof(Span<Int64,DynExtent>) = " << sizeof(Span<Int64,DynExtent>) << "\n";

  // Vérifie que [[no_unique_address]] est bien pris en compte
  ASSERT_EQ(sizeof(Span<Int32,1>),sizeof(void*));
  ASSERT_EQ(sizeof(Span<Int64, 1>), sizeof(void*));

  std::array<Int64, 12> vlist{ 9, 13, 32, 27, 43, -5, 2, -7, 8, 11, 25, 48 };
  Span<Int64, 12> fixed_list(vlist);
  Int32 sub_view_size = 5;
  Span<Int64> sub_view1 = fixed_list.subspan(0, sub_view_size);
  ASSERT_EQ(sub_view1.size(),sub_view_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
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

template class Span<Int32,DynExtent,-1>;
template class Span<const Int32,DynExtent,-1>;
template class Span<double,DynExtent,-1>;
template class Span<const double,DynExtent,-1>;

template class SmallSpan<Int32,DynExtent,-1>;
template class SmallSpan<const Int32,DynExtent,-1>;
template class SmallSpan<double,DynExtent,-1>;
template class SmallSpan<const double,DynExtent,-1>;

template class Span<Int32,4>;
template class Span<const Int32,5>;
template class Span<double,6>;
template class Span<const double,7>;

template class SmallSpan<Int32,4>;
template class SmallSpan<const Int32,5>;
template class SmallSpan<double,6>;
template class SmallSpan<const double,7>;

template class Span<Int32,4,-1>;
template class Span<const Int32,5,-1>;
template class Span<double,6,-1>;
template class Span<const double,7,-1>;

template class SmallSpan<Int32,4,-1>;
template class SmallSpan<const Int32,5,-1>;
template class SmallSpan<double,6,-1>;
template class SmallSpan<const double,7,-1>;

template class Span2<Int32>;
template class Span2<const Int32>;
template class Span2<double>;
template class Span2<const double>;

template class SmallSpan2<Int32>;
template class SmallSpan2<const Int32>;
template class SmallSpan2<double>;
template class SmallSpan2<const double>;
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
