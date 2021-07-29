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

#include "arccore/collections/Array.h"
#include "arccore/collections/Array2.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Iterator.h"

using namespace Arccore;

namespace
{
class IntSubClass
{
 public:
  IntSubClass(Integer v) : m_v(v) {}
  IntSubClass() : m_v(0) {}
  Integer m_v;
  bool operator==(Integer iv) const { return m_v==iv; }
};
}
namespace Arccore
{
ARCCORE_DEFINE_ARRAY_PODTYPE(IntSubClass);
}
namespace
{
void
_testArraySwap(bool use_own_swap)
{
  std::cout << "** TestArraySwap is_own=" << use_own_swap << "\n";

  UniqueArray<IntSubClass> c1(7);
  IntSubClass* x1 = c1.unguardedBasePointer();
  std::cout << "** C1_this = " << &c1 << "\n";
  std::cout << "** C1_BASE = " << x1 << "\n";
  UniqueArray<IntSubClass> c2(3);
  IntSubClass* x2 = c2.unguardedBasePointer();
  std::cout << "** C2_this = " << &c2 << "\n";
  std::cout << "** C2_BASE = " << x2 << "\n";

  if (use_own_swap){
    swap(c1,c2);
  }
  else
    std::swap(c1,c2);

  IntSubClass* after_x1 = c1.data();
  IntSubClass* after_x2 = c2.data();
  std::cout << "** C1_BASE_AFTER = " << after_x1 << " size=" << c1.size() << "\n";
  std::cout << "** C2_BASE_AFTER = " << after_x2 << " size=" << c2.size() << "\n";

  ASSERT_TRUE(x1==after_x2) << "Bad value after swap [1]";
  ASSERT_TRUE(x2==after_x1) << "Bad value after swap [2]";
}
}

TEST(Array, Swap1)
{
  _testArraySwap(true);
}

TEST(Array, Swap2)
{
  _testArraySwap(false);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_UT_CHECK(expr,message) \
if ( ! (expr) )\
  throw Arccore::FatalErrorException((message))

class IntPtrSubClass
{
public:
  static Integer count;
  IntPtrSubClass(Integer v) : m_v(new Integer(v)) { ++count; }
  IntPtrSubClass() : m_v(new Integer(0)) { ++count; }
  ~IntPtrSubClass() { --count; delete m_v; }
  Integer* m_v;
  IntPtrSubClass(const IntPtrSubClass& v) : m_v(new Integer(*v.m_v)) { ++count; }
  void operator=(const IntPtrSubClass& v)
    {
      Integer* n = new Integer(*v.m_v);
      delete m_v;
      m_v = n;
    }
  bool operator==(Integer iv) const
  {
    //cout << "** COMPARE: " << *m_v << " v=" << iv << '\n';
    return *m_v==iv;
  }
};
Integer IntPtrSubClass::count = 0;

template<typename Container,typename SubClass>
class IntegerArrayTester
{
public:
  void test()
  {
    {
      Container c;
      c.add(SubClass(1));
      c.add(SubClass(2));
      c.add(SubClass(3));
      ARCCORE_UT_CHECK((c.size()==3),"Bad size (3)");
      ARCCORE_UT_CHECK((c[0]==1),"Bad value [0]");
      ARCCORE_UT_CHECK((c[1]==2),"Bad value [1]");
      ARCCORE_UT_CHECK((c[2]==3),"Bad value [2]");
      c.resize(0);
      ARCCORE_UT_CHECK((c.size()==0),"Bad size (0)");
      c.resize(5);
      ARCCORE_UT_CHECK((c.size()==5),"Bad size");
      c.add(SubClass(6));
      ARCCORE_UT_CHECK((c.size()==6),"Bad size");
      ARCCORE_UT_CHECK((c[5]==6),"Bad value [5]");
      c.shrink();
      ASSERT_EQ(c.size(),c.capacity()) << "Bad capacity (test 1)";
      c.resize(12);
      c.shrink();
      ASSERT_EQ(c.size(),c.capacity()) << "Bad capacity (test 2)";
    }
    {
      Container c;
      c.shrink();
      ASSERT_EQ(c.capacity(),0) << "Bad capacity (test 3)";
    }
    {
      Container c;
      Integer nb = 20;
      for( Integer i=0; i<nb; ++i )
        c.add(SubClass(i));
      c.reserve(nb*2);
      Int64 current_capacity = c.capacity();
      ASSERT_EQ(current_capacity,(nb*2)) << "Bad capacity (test 4)";
      c.shrink(c.capacity()+5);
      ASSERT_EQ(c.capacity(),current_capacity) << "Bad capacity (test 5)";
      c.shrink(32);
      ASSERT_EQ(c.capacity(),32) << "Bad capacity (test 6)";
      c.shrink();
      ASSERT_EQ(c.capacity(),c.size()) << "Bad capacity (test 7)";
    }
    {
      UniqueArray<Container> uc;
      Integer nb = 1000000;
      uc.resize(1000);
      for( Container& c : uc ){
        c.reserve(nb);
        c.shrink_to_fit();
      }
    }
    {
      Container c;
      for( Integer i=0; i<50; ++i )
        c.add(SubClass(i+2));
      {
        Container c2 = c;
        for( Integer i=50; i<100; ++i ){
          c2.add(SubClass(i+2));
        }
        Container c4 = c2;
        Container c3;
        c3.add(5);
        c3.add(6);
        c3.add(7);
        c3 = c2;
        for( Integer i=100; i<150; ++i ){
          c4.add(SubClass(i+2));
        }
      }
      ARCCORE_UT_CHECK((c.size()==150),"Bad size (150)");
      c.reserve(300);
      ARCCORE_UT_CHECK((c.capacity()==300),"Bad capacity (300)");
      for( Integer i=0; i<50; ++i ){
        c.remove(i);
      }
      ARCCORE_UT_CHECK((c.size()==100),"Bad size (100)");
      for( Integer i=0; i<50; ++i ){
        //cout << "** VAL: " << i << " c=" << c[i] << " expected=" << ((i*2)+3) << '\n';
        ARCCORE_UT_CHECK((c[i]==((i*2)+3)),"Bad value");
      }
      for( Integer i=50; i<100; ++i ){
        //cout << "** VAL: " << i << " c=" << c[i] << " expected=" << (i+52) << '\n';
        ARCCORE_UT_CHECK((c[i]==(i+52)),"Bad value");
      }
    }
  }
};

namespace
{
void
_testArrayNewInternal()
{
  using namespace Arccore;
  std::cout << "** TEST VECTOR NEW\n";

  size_t impl_size = sizeof(ArrayImplBase);
  size_t wanted_size = AlignedMemoryAllocator::simdAlignment();
  std::cout << "** sizeof(ArrayImplBase) = " << impl_size << '\n';
  if (impl_size!=wanted_size)
    ARCCORE_FATAL("Bad sizeof(ArrayImplBase) v={0} expected={1}",impl_size,wanted_size);
  {
    IntegerArrayTester< SharedArray<IntSubClass>, IntSubClass > rvt;
    rvt.test();
  }
  std::cout << "** TEST VECTOR NEW 2\n";
  {
    IntegerArrayTester< SharedArray<IntPtrSubClass>, IntPtrSubClass > rvt;
    rvt.test();
    std::cout << "** COUNT = " << IntPtrSubClass::count << "\n";

  }
  std::cout << "** TEST VECTOR NEW 3\n";
  {
    IntegerArrayTester< SharedArray<IntPtrSubClass>, IntPtrSubClass > rvt;
    rvt.test();
    std::cout << "** COUNT = " << IntPtrSubClass::count << "\n";

  }
  {
    SharedArray<IntSubClass> c;
    c.add(5);
    c.add(7);
    SharedArray<IntSubClass> c2(c.clone());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c;
    c.add(5);
    c.add(7);
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c { 5, 7 };
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c { 5, 7 };
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
  }
  {
    SharedArray<IntSubClass> c { 5, 7 };
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
  }
  {
    PrintableMemoryAllocator allocator;
    UniqueArray<IntSubClass> c(&allocator);
    UniqueArray<IntSubClass> cx(&allocator,5);
    ARCCORE_UT_CHECK((cx.size()==5),"Bad value [5]");
    c.add(5);
    c.add(7);
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c.size()==2),"Bad value [2]");
    ARCCORE_UT_CHECK((c[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
    for( Integer i=0; i<50; ++i )
      c.add(i+3);
    c.resize(24);
    c.reserve(70);
  }
  {
    SharedArray<IntSubClass> c2;
    c2.add(3);
    c2.add(5);
    {
      SharedArray<IntSubClass> c;
      c.add(5);
      c.add(7);
      c2 = c;
    }
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size()==3),"Bad value [3]");
    ARCCORE_UT_CHECK((c2[0]==5),"Bad value [5]");
    ARCCORE_UT_CHECK((c2[1]==7),"Bad value [7]");
    ARCCORE_UT_CHECK((c2[2]==3),"Bad value [7]");
  }
  {
    UniqueArray<Int32> values1 = { 2, 5 };
    UniqueArray<Int32> values2 = { 4, 9, 7 };
    // Copie les valeurs de values2 Ã  la fin de values1.
    std::copy(std::begin(values2),std::end(values2),std::back_inserter(values1));
    std::cout << "** VALUES1 = " << values1 << "\n";
    ARCCORE_UT_CHECK((values1.size()==5),"BI: Bad size");
    ARCCORE_UT_CHECK((values1[0]==2),"BI: Bad value [0]");
    ARCCORE_UT_CHECK((values1[1]==5),"BI: Bad value [1]");
    ARCCORE_UT_CHECK((values1[2]==4),"BI: Bad value [2]");
    ARCCORE_UT_CHECK((values1[3]==9),"BI: Bad value [3]");
    ARCCORE_UT_CHECK((values1[4]==7),"BI: Bad value [4]");

    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << '\n';
  }
  {
    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    [[maybe_unused]] auto range = vx.range();
    UniqueArray<IntPtrSubClass>::iterator i = vx.begin();
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << '\n';
    const UniqueArray<IntPtrSubClass>& cvx = (const UniqueArray<IntPtrSubClass>&)(vx);
    UniqueArray<IntPtrSubClass>::const_iterator cicvx = cvx.begin();
    std::cout << "V=" << cicvx->m_v << '\n';
  }
  {
    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    [[maybe_unused]] auto r = vx.range();
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::iterator iend = std::end(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << " " << (iend-i) << '\n';
    const UniqueArray<IntPtrSubClass>& cvx = (const UniqueArray<IntPtrSubClass>&)(vx);
    UniqueArray<IntPtrSubClass>::const_iterator cicvx = std::begin(cvx);
    std::cout << "V=" << cicvx->m_v << '\n';
    std::copy(std::begin(vx),std::end(vx),std::begin(vx));
  }
  {
    UniqueArray<Int32> values = { 4, 9, 7 };
    for( typename ArrayView<Int32>::const_iter i(values.view()); i(); ++i ){
      std::cout << *i << '\n';
    }
    for( typename ConstArrayView<Int32>::const_iter i(values.view()); i(); ++i ){
      std::cout << *i << '\n';
    }
    for( auto i : values.range()){
      std::cout << i << '\n';
    }
    for( auto i : values.constView().range()) {
      std::cout << i << '\n';
    }

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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc)
{
  _testArrayNewInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void
_Add(Array<Real>& v,Integer new_size)
{
  v.resize(new_size);
}
}

TEST(Array, Misc2)
{
  using namespace Arccore;
  {
    UniqueArray<Real> v;
    v.resize(3);
    v.add(4.3);
    for( Integer i=0, is=v.size(); i<is; ++i ){
      std::cout << " Value: " << v[i] << '\n';
    }
    v.printInfos(std::cout);
  }
  {
    UniqueArray<Real> v;
    v.add(4.3);
    v.add(5.3);
    v.add(4.6);
    v.printInfos(std::cout);
    v.add(14.3);
    v.add(25.3);
    v.add(34.6);
    v.printInfos(std::cout);
    v.resize(4);
    v.printInfos(std::cout);
    v.add(12.2);
    v.add(2.4);
    v.add(3.4);
    v.add(3.6);
    v.printInfos(std::cout);
    for( Integer i=0, is=v.size(); i<is; ++i ){
      std::cout << " Value: " << v[i] << '\n';
    }
  }
  {
    UniqueArray<Real> v;
    v.reserve(5);
    for( int i=0; i<10; ++i )
      v.add((Real)i);
    for( Integer i=0, is=v.size(); i<is; ++i ){
      std::cout << " Value: " << v[i] << '\n';
    }
  }
  {
    UniqueArray<Real> v;
    v.reserve(175);
    for( int i=0; i<27500; ++i ){
      Real z = (Real)i;
      v.add(z*z);
    }
    for( int i=0; i<5000; ++i ){
      v.remove(i*2);
    }
    v.reserve(150);
    for( int i=0; i<27500; ++i ){
      Real z = (Real)i;
      v.add(z*z);
    }
    std::cout << " ValueSize= " << v.size() << " values=" << v << '\n';
  }
  for( Integer i=0; i<100; ++i ){
    UniqueArray<Real> v;
    _Add(v,500000);
    _Add(v,1000000);
    _Add(v,0);
    _Add(v,100000);
    _Add(v,0);
    _Add(v,0);
    _Add(v,230000);
    std::cout << " Size: " << v.size() << '\n';
  }
}

TEST(Array2, Misc)
{
  using namespace Arccore;

  {
    UniqueArray2<Int32> c;
    c.resize(3,5);
    Integer nb = 15;
    c.reserve(nb*2);
    Int64 current_capacity = c.capacity();
    ASSERT_EQ(current_capacity,(nb*2)) << "Bad capacity (test 1)";
    c.shrink(32);
    ASSERT_EQ(c.capacity(),current_capacity) << "Bad capacity (test 2)";
    c.shrink();
    c.shrink_to_fit();
    ASSERT_EQ(c.capacity(),c.totalNbElement()) << "Bad capacity (test 3)";
  }
}
