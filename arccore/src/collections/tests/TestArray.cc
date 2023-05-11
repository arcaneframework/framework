// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
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
  bool operator==(const IntSubClass& v) const { return m_v==v.m_v; }
};
class IntSubClassNoPod
{
 public:
  explicit IntSubClassNoPod(Integer v) : m_v(v) {}
  //IntSubClassNoPod() : m_v(0) {}
  Integer m_v;
  friend bool operator==(const IntSubClassNoPod& v,Integer iv) { return v.m_v == iv; }
  friend bool operator==(const IntSubClassNoPod& v1,const IntSubClassNoPod& v2) { return v1.m_v==v2.m_v; }
  friend bool operator!=(const IntSubClassNoPod& v1,const IntSubClassNoPod& v2) { return v1.m_v!=v2.m_v; }
  friend std::ostream& operator<<(std::ostream& o,IntSubClassNoPod c)
  {
    o << c.m_v;
    return o;
  }
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
  std::cout << "** wanted_size = " << AlignedMemoryAllocator::simdAlignment() << "\n";
  std::cout << "** sizeof(ArrayImplBase) = " << impl_size << '\n';
  //if (impl_size!=wanted_size)
  //ARCCORE_FATAL("Bad sizeof(ArrayImplBase) v={0} expected={1}",impl_size,wanted_size);
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
    // Copie les valeurs de values2 à la fin de values1.
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
    UniqueArray<Int32> values1;
    UniqueArray<Int32> values2 = { 4, 9, 7, 6, 3 };
    // Copie les valeurs de values2 à la fin de values1.
    values1.copy(values2);
    std::cout << "** VALUES1 = " << values1 << "\n";
    ARCCORE_UT_CHECK((values1.size()==5),"BI: Bad size");
    ARCCORE_UT_CHECK((values1[0]==4),"BI2: Bad value [0]");
    ARCCORE_UT_CHECK((values1[1]==9),"BI2: Bad value [1]");
    ARCCORE_UT_CHECK((values1[2]==7),"BI2: Bad value [2]");
    ARCCORE_UT_CHECK((values1[3]==6),"BI2: Bad value [3]");
    ARCCORE_UT_CHECK((values1[4]==3),"BI2: Bad value [4]");

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
    {
      UniqueArray<IntSubClassNoPod> c{ IntSubClassNoPod{ 5 }, IntSubClassNoPod{ 7 } };
      UniqueArray<IntSubClassNoPod> c2(c.constView());
      c2.add(IntSubClassNoPod{ 3 });
      ARCCORE_UT_CHECK((c2.size() == 3), "Bad value [3]");
      ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
      ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
      ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
      ARCCORE_UT_CHECK((c2[0] == 5), "Bad value [5]");
      ARCCORE_UT_CHECK((c2[1] == 7), "Bad value [7]");
      ARCCORE_UT_CHECK((c2[2] == 3), "Bad value [7]");
      c = c2.span();
      ASSERT_EQ(c.constSpan(),c2.constSpan());
    }
  }
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc)
{
  try{
    _testArrayNewInternal();
  }
  catch(const Exception& ex){
    std::cerr << "Exception ex=" << ex << "\n";
    throw;
  }
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc2)
{
  using namespace Arccore;
  {
    UniqueArray<Real> v;
    v.resize(3);
    v.add(4.3);
    for (Integer i = 0, is = v.size(); i < is; ++i) {
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
    for (Integer i = 0, is = v.size(); i < is; ++i) {
      std::cout << " Value: " << v[i] << '\n';
    }
  }
  {
    UniqueArray<Real> v;
    v.reserve(5);
    for (int i = 0; i < 10; ++i)
      v.add((Real)i);
    for (Integer i = 0, is = v.size(); i < is; ++i) {
      std::cout << " Value: " << v[i] << '\n';
    }
  }
  {
    UniqueArray<Real> v;
    v.reserve(175);
    for (int i = 0; i < 27500; ++i) {
      Real z = (Real)i;
      v.add(z * z);
    }
    for (int i = 0; i < 5000; ++i) {
      v.remove(i * 2);
    }
    v.reserve(150);
    for (int i = 0; i < 27500; ++i) {
      Real z = (Real)i;
      v.add(z * z);
    }
    std::cout << " ValueSize= " << v.size() << " values=" << v << '\n';
  }
  for (Integer i = 0; i < 100; ++i) {
    UniqueArray<Real> v;
    _Add(v, 500000);
    _Add(v, 1000000);
    _Add(v, 0);
    _Add(v, 100000);
    _Add(v, 0);
    _Add(v, 0);
    _Add(v, 230000);
    std::cout << " Size: " << v.size() << '\n';
    ASSERT_EQ(v.size(), 230000);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array2, Misc)
{
  using namespace Arccore;

  {
    UniqueArray2<Int32> c;
    c.resize(3, 5);
    Integer nb = 15;
    c.reserve(nb * 2);
    Int64 current_capacity = c.capacity();
    ASSERT_EQ(current_capacity, (nb * 2)) << "Bad capacity (test 1)";
    c.shrink(32);
    ASSERT_EQ(c.capacity(), current_capacity) << "Bad capacity (test 2)";
    c.shrink();
    c.shrink_to_fit();
    ASSERT_EQ(c.capacity(), c.totalNbElement()) << "Bad capacity (test 3)";
    ASSERT_EQ(c[1][2], c(1, 2));
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
    bool is_ok = c[2, 1] == c(2, 1);
    ASSERT_TRUE(is_ok);
#endif
  }
  {
    UniqueArray2<Int32> c;
    c.resize(2, 1);
    std::cout << "V1=" << c.to1DSpan() << "\n";
    c[0][0] = 2;
    c[1][0] = 3;
    c.resize(2, 2);
    std::cout << "V2=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 2);
    ASSERT_EQ(c[1][0], 3);
    ASSERT_EQ(c[0][1], 0);
    ASSERT_EQ(c[1][1], 0);
  }
  {
    UniqueArray2<Int32> c;
    c.resizeNoInit(2, 1);
    c[0][0] = 1;
    c[1][0] = 2;
    c.resizeNoInit(2, 2);
    c[0][1] = 4;
    c[1][1] = 5;
    std::cout << "X1=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 1);
    ASSERT_EQ(c[1][0], 2);
    ASSERT_EQ(c[0][1], 4);
    ASSERT_EQ(c[1][1], 5);
    c.resize(3, 2);
    std::cout << "X2=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 1);
    ASSERT_EQ(c[1][0], 2);
    ASSERT_EQ(c[0][1], 4);
    ASSERT_EQ(c[1][1], 5);
    ASSERT_EQ(c[2][0], 0);
    ASSERT_EQ(c[2][1], 0);
    c[2][0] = 8;
    c[2][1] = 10;
    c.resize(6, 5);
    std::cout << "X3=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 1);
    ASSERT_EQ(c[1][0], 2);
    ASSERT_EQ(c[0][1], 4);
    ASSERT_EQ(c[1][1], 5);
    ASSERT_EQ(c[2][0], 8);
    ASSERT_EQ(c[2][1], 10);
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(c[i][2], 0);
      ASSERT_EQ(c[i][3], 0);
      ASSERT_EQ(c[i][4], 0);
    }
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(c[3][j], 0);
      ASSERT_EQ(c[4][j], 0);
      ASSERT_EQ(c[5][j], 0);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, SubViews)
{
  using namespace Arccore;

  {
    // Test Array::subView() et Array::subConstView()
    UniqueArray<Int32> v;
    v.resize(23);
    for (Int32 i = 0, n = v.size(); i < n; ++i)
      v[i] = (i + 1);

    auto sub_view1 = v.subView(50, 5);
    ASSERT_EQ(sub_view1.data(), nullptr);
    ASSERT_EQ(sub_view1.size(), 0);

    auto sub_view2 = v.subView(2, 8);
    ASSERT_EQ(sub_view2.size(), 8);
    for (Int32 i = 0, n = sub_view2.size(); i < n; ++i)
      ASSERT_EQ(sub_view2[i], v[2 + i]);

    auto sub_view3 = v.subView(20, 8);
    ASSERT_EQ(sub_view3.size(), 3);
    for (Int32 i = 0, n = sub_view3.size(); i < n; ++i)
      ASSERT_EQ(sub_view3[i], v[20 + i]);

    auto sub_const_view1 = v.subConstView(50, 5);
    ASSERT_EQ(sub_const_view1.data(), nullptr);
    ASSERT_EQ(sub_const_view1.size(), 0);

    auto sub_const_view2 = v.subConstView(2, 8);
    ASSERT_EQ(sub_const_view2.size(), 8);
    for (Int32 i = 0, n = sub_const_view2.size(); i < n; ++i)
      ASSERT_EQ(sub_const_view2[i], v[2 + i]);

    auto sub_const_view3 = v.subConstView(20, 8);
    ASSERT_EQ(sub_const_view3.size(), 3);
    for (Int32 i = 0, n = sub_const_view3.size(); i < n; ++i)
      ASSERT_EQ(sub_const_view3[i], v[20 + i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class MyArrayTest
: public Arccore::Array<DataType>
{
 public:

  using BaseClass = Arccore::Array<DataType>;
  using BaseClass::_resizeNoInit;

 public:

  void resizeNoInit(Int64 new_size)
  {
    _resizeNoInit(new_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc3)
{
  using namespace Arccore;

  {
    MyArrayTest<IntSubClassNoPod> c;
    const Int32 ref_value1 = 12;
    const Int32 ref_value2 = 7;

    c.resize(9,IntSubClassNoPod(ref_value1));
    std::cout << "C1=" << c << "\n";
    for(IntSubClassNoPod x : c )
      ASSERT_EQ(x,ref_value1);

    c.resize(21,IntSubClassNoPod(ref_value2));
    ASSERT_EQ(c.size(),21);
    std::cout << "C2=" << c << "\n";

    // Redimensionne sans initialiser. Les valeurs pour les éléments
    // de 9 à 18 doivent valoir \a ref_value2
    c.resizeNoInit(18);
    std::cout << "C4=" << c << "\n";
    for( Int32 i=9, s=c.size(); i<s; ++i )
      ASSERT_EQ(c[i],ref_value2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc4)
{
  using namespace Arccore;

  IMemoryAllocator* allocator1 = AlignedMemoryAllocator::Simd();
  PrintableMemoryAllocator printable_allocator2;
  IMemoryAllocator* allocator2 = &printable_allocator2;

  {
    std::cout << "Array a\n";
    UniqueArray<Int32> a(allocator1);
    ASSERT_EQ(a.allocator(),allocator1);
    a.add(27);
    a.add(38);
    a.add(13);
    a.add(-5);

    std::cout << "Array b\n";
    UniqueArray<Int32> b(allocator2);
    ASSERT_EQ(b.capacity(),0);
    ASSERT_EQ(b.size(),0);
    ASSERT_EQ(b.allocator(),allocator2);

    b = a;
    ASSERT_EQ(b.size(),a.size());
    ASSERT_EQ(b.allocator(),a.allocator());

    std::cout << "Array c\n";
    UniqueArray<Int32> c(a.clone());
    ASSERT_EQ(c.allocator(),a.allocator());
    ASSERT_EQ(c.size(),a.size());
    ASSERT_EQ(c.constSpan(),a.constSpan());

    std::cout << "Array d\n";
    UniqueArray<Int32> d(allocator2,a);
    ASSERT_EQ(d.allocator(),allocator2);
    ASSERT_EQ(d.size(),a.size());
    ASSERT_EQ(d.constSpan(),a.constSpan());

    std::cout << "Array e\n";
    UniqueArray<Int32> e(allocator2,25);
    ASSERT_EQ(e.allocator(),allocator2);
    ASSERT_EQ(e.size(),25);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Allocator)
{
  using namespace Arccore;
  PrintableMemoryAllocator printable_allocator;
  PrintableMemoryAllocator printable_allocator2;
  IMemoryAllocator* allocator1 = AlignedMemoryAllocator::Simd();
  IMemoryAllocator* allocator2 = &printable_allocator;
  {
    std::cout << "Array a1\n";
    UniqueArray<Int32> a1(allocator2);
    ASSERT_EQ(a1.allocator(), allocator2);
    ASSERT_EQ(a1.size(), 0);
    ASSERT_EQ(a1.capacity(), 0);
    ASSERT_EQ(a1.data(), nullptr);

    std::cout << "Array a2\n";
    UniqueArray<Int32> a2(a1);
    ASSERT_EQ(a1.allocator(), a2.allocator());
    ASSERT_EQ(a2.capacity(), 0);
    ASSERT_EQ(a2.data(), nullptr);
    std::array<Int32, 5> vals = { 5, 7, 12, 3, 1 };
    a1.reserve(3);
    a1.add(5);
    a1.add(7);
    a1.add(12);
    a1.add(3);
    a1.add(1);
    ASSERT_EQ(a1.size(), 5);

    std::cout << "Array a3\n";
    UniqueArray<Int32> a3(allocator1);
    a3.add(4);
    a3.add(6);
    a3.add(2);
    ASSERT_EQ(a3.size(), 3);
    a3 = a1;
    ASSERT_EQ(a3.allocator(), a1.allocator());
    ASSERT_EQ(a3.size(), a1.size());
    ASSERT_EQ(a3.constSpan(), a1.constSpan());

    std::cout << "Array a4\n";
    UniqueArray<Int32> a4(allocator1);
    a4.add(4);
    a4.add(6);
    a4.add(2);
    ASSERT_EQ(a4.size(), 3);
    a4 = a1.span();
    ASSERT_EQ(a4.allocator(), allocator1);

    a4 = UniqueArray<Int32>(&printable_allocator2);

    UniqueArray<Int32> array[2];
    IMemoryAllocator* allocator3 = allocator1;
    for( Integer i=0; i<2; ++i ){
      array[i] = UniqueArray<Int32>(allocator3);
    }
    ASSERT_EQ(array[0].allocator(), allocator3);
    ASSERT_EQ(array[1].allocator(), allocator3);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
// Instancie explicitement les classes tableaux pour garantir
// que toutes les méthodes fonctionnent
template class UniqueArray<IntSubClass>;
template class SharedArray<IntSubClass>;
template class Array<IntSubClass>;
template class AbstractArray<IntSubClass>;
template class UniqueArray2<IntSubClass>;
template class SharedArray2<IntSubClass>;
template class Array2<IntSubClass>;
}
