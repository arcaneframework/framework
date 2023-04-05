// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"

#include "arcane/utils/SmallArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(Collections,Basic)
{
  std::cout << "TEST_Collection Basic\n";

  StringList string_list;
  String str1 = "TotoTiti";
  String str2 = "Tata";
  String str3 = "Hello";
  String str4 = "MyStringToTest";
  
  string_list.add(str1);
  ASSERT_EQ(string_list.count(),1);

  string_list.add(str2);
  ASSERT_EQ(string_list.count(),2);

  string_list.add(str3);
  ASSERT_EQ(string_list.count(),3);

  ASSERT_TRUE(string_list.contains("Tata"));
  ASSERT_FALSE(string_list.contains("NotTata"));
  ASSERT_EQ(string_list[0],str1);
  ASSERT_EQ(string_list[1],"Tata");
  ASSERT_EQ(string_list[2],str3);

  string_list.remove("Tata");
  ASSERT_EQ(string_list.count(),2);
  ASSERT_EQ(string_list[0],str1);
  ASSERT_EQ(string_list[1],str3);

  string_list.clear();
  ASSERT_EQ(string_list.count(),0);

  string_list.add(str4);
  ASSERT_EQ(string_list.count(),1);
  string_list.add(str2);
  ASSERT_EQ(string_list.count(),2);
  string_list.add(str1);
  ASSERT_EQ(string_list.count(),3);

  ASSERT_TRUE(string_list.contains("Tata"));
  ASSERT_FALSE(string_list.contains("NotTata"));
  ASSERT_TRUE(string_list.contains(str2));
  ASSERT_FALSE(string_list.contains(str3));
  ASSERT_TRUE(string_list.contains(str1));
}

void
_checkSmallArrayValues(Span<const Int32> view)
{
  for( Int64 i=0, n=view.size(); i<n; ++i )
    ASSERT_EQ(view[i],i+1);
}

void
_checkSmallArrayValues(Span<const Int32> view1,Span<const Int32> view2)
{
  Int64 n1 = view1.size();
  Int64 n2 = view2.size();
  ASSERT_EQ(n1,n2);
  for( Int64 i=0; i<n1; ++i )
    ASSERT_EQ(view1[i],view2[i]);
}

TEST(Collections,SmallArray)
{
  {
    constexpr int N = 934;
    char buf[N];
    impl::StackMemoryAllocator b(buf,N);
    ASSERT_EQ(b.guarantedAlignment(),0);
  }
  {
    SmallArray<Int32,400> buf1;
    for( Int32 i=0; i<200; ++i )
      buf1.add(i+1);
    ASSERT_EQ(buf1.size(),200);
    _checkSmallArrayValues(buf1);

    buf1.resize(50);
    buf1.shrink();
    ASSERT_EQ(buf1.size(),50);
    _checkSmallArrayValues(buf1);

    for( Int32 i=0; i<200; ++i )
      buf1.add(50+i+1);
    ASSERT_EQ(buf1.size(),250);
    _checkSmallArrayValues(buf1);
  }
  for( int z=1; z<10; ++z ) {
    UniqueArray<Int32> ref_buf(250*z);
    for(Int32 i=0, n=ref_buf.size(); i<n; ++i )
      ref_buf[i] = (i+1)*2;

    UniqueArray<Int32> ref_buf2(100*z*z);
    for(Int32 i=0, n=ref_buf2.size(); i<n; ++i )
      ref_buf2[i] = (i+13)*3;

    SmallArray<Int32,1024> buf2(ref_buf);
    _checkSmallArrayValues(buf2,ref_buf);
    SmallArray<Int32,1024> buf3(ref_buf.span());
    _checkSmallArrayValues(buf3,ref_buf);
    SmallArray<Int32,1024> buf4(ref_buf.constSpan());
    _checkSmallArrayValues(buf4,ref_buf);
    SmallArray<Int32,1024> buf5(ref_buf.view());
    _checkSmallArrayValues(buf5,ref_buf);
    SmallArray<Int32,1024> buf6(ref_buf.constView());
    _checkSmallArrayValues(buf6,ref_buf);

    buf2 = ref_buf2;
    _checkSmallArrayValues(buf2,ref_buf2);
    buf3 = ref_buf2.span();
    _checkSmallArrayValues(buf3,ref_buf2);
    buf4 = ref_buf2.constSpan();
    _checkSmallArrayValues(buf4,ref_buf2);
    buf5 = ref_buf2.view();
    _checkSmallArrayValues(buf5,ref_buf2);
    buf6 = ref_buf2.constView();
    _checkSmallArrayValues(buf6,ref_buf2);
  }
  {
    for( int z=1; z<10; ++z ) {
      Int32 n = 5+(z*100);
      SmallArray<Int32> buf3(n);
      ASSERT_EQ(buf3.size(),n);
      for(Int32 i=0; i<n; ++i )
        buf3[i] = (i*22)+1;
      for(Int32 i=0; i<n; ++i )
        ASSERT_EQ(buf3[i],((i*22)+1));
    }
  }
  {
    std::cout << "Test initializer_list 1\n";
    SmallArray<Int32,20> buf = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,
      27, 29, 31, 33, 35, 37, 39, 41 };
    Int32 n = 21;
    ASSERT_EQ(buf.size(),n);
    for(Int32 i=0; i<n; ++i )
      ASSERT_EQ(buf[i],((i*2)+1));
  }
  {
    std::cout << "Test initializer_list 2\n";
    SmallArray<Int32,100> buf = { 1, 3, 5, 7, 9, 11 };
    Int32 n = 6;
    ASSERT_EQ(buf.size(),n);
    for(Int32 i=0; i<n; ++i )
      ASSERT_EQ(buf[i],((i*2)+1));
  }
  {
    size_t s1 = 513;
    SmallArray<Int32> buf1(s1);
    ASSERT_EQ(buf1.size(),s1);

    Int64 s2 = 217;
    SmallArray<Int32> buf2(s2);
    ASSERT_EQ(buf2.size(),s2);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template class List<String>;
template class ListImplBase<String>;
template class ListImplT<String>;
template class Collection<String>;
template class CollectionImplT<String>;

template class SmallArray<Int32>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
