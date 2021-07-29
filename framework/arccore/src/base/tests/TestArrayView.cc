// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
