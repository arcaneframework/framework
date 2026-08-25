// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/collections/Array.h"
#include "arccore/collections/IMemoryAllocator.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Iterator.h"

#include "TestArrayCommon.h"

#include <iostream>

using namespace Arccore;
using namespace TestArccore;

namespace
{
void _testArraySwap(bool use_own_swap)
{
  std::cout << "** TestArraySwap is_own=" << use_own_swap << "\n";

  String c1_name = "TestC1";
  UniqueArray<IntSubClass> c1(7);
  c1.setDebugName(c1_name);
  IntSubClass* x1 = c1.data();
  std::cout << "** C1_this = " << &c1 << "\n";
  std::cout << "** C1_BASE = " << x1 << "\n";
  UniqueArray<IntSubClass> c2(3);
  IntSubClass* x2 = c2.data();
  std::cout << "** C2_this = " << &c2 << "\n";
  std::cout << "** C2_BASE = " << x2 << "\n";

  ASSERT_EQ(c1.debugName(), c1_name);
  ASSERT_EQ(c2.debugName(), String{});

  if (use_own_swap) {
    swap(c1, c2);
  }
  else
    std::swap(c1, c2);

  ASSERT_EQ(c2.debugName(), c1_name);
  ASSERT_EQ(c1.debugName(), String{});

  IntSubClass* after_x1 = c1.data();
  IntSubClass* after_x2 = c2.data();
  std::cout << "** C1_BASE_AFTER = " << after_x1 << " size=" << c1.size() << "\n";
  std::cout << "** C2_BASE_AFTER = " << after_x2 << " size=" << c2.size() << "\n";

  ASSERT_TRUE(x1 == after_x2) << "Bad value after swap [1]";
  ASSERT_TRUE(x2 == after_x1) << "Bad value after swap [2]";
}
} // namespace

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

namespace
{
template <typename ArrayType>
void _testArrayAssign()
{
  using namespace Arccore;
  using value_type = ArrayType::value_type;
  // Test assign(count, val)
  {
    ArrayType a;
    a.assign(5l, value_type(42));
    ASSERT_EQ(a.size(), 5);
    for (Int32 i = 0; i < 5; ++i)
      ASSERT_EQ(a[i], 42);

    // Test assign with count = 0
    a.assign(0l, value_type(99));
    ASSERT_EQ(a.size(), 0);

    // Test assign overwriting existing data
    a.assign(3l, value_type(1));
    a.assign(2l, value_type(7));
    ASSERT_EQ(a.size(), 2);
    ASSERT_EQ(a[0], 7);
    ASSERT_EQ(a[1], 7);
  }

  // Test assign(InputIterator first, InputIterator last)
  {
    ArrayType a;
    std::vector<Int32> src = { 1, 2, 3, 4, 5 };
    a.assign(src.begin(), src.end());
    ASSERT_EQ(a.size(), 5);
    for (Int32 i = 0; i < 5; ++i)
      ASSERT_EQ(a[i], i + 1);

    // Test with empty range
    ArrayType a2;
    std::vector<Int32> empty;
    a2.assign(empty.begin(), empty.end());
    ASSERT_EQ(a2.size(), 0);

    // Test with array iterators
    ArrayType a3;
    ArrayType src_array = { 10, 20, 30 };
    a3.assign(src_array.begin(), src_array.end());
    ASSERT_EQ(a3.size(), 3);
    ASSERT_EQ(a3[0], 10);
    ASSERT_EQ(a3[1], 20);
    ASSERT_EQ(a3[2], 30);

    // Test with initializer_list (via vector)
    ArrayType a4;
    std::vector<Int32> il_src = { 100, 200 };
    a4.assign(il_src.begin(), il_src.end());
    ASSERT_EQ(a4.size(), 2);
    ASSERT_EQ(a4[0], 100);
    ASSERT_EQ(a4[1], 200);
  }
}

template <typename ArrayType>
void _testArrayErase()
{
  using namespace Arccore;
  using value_type = ArrayType::value_type;

  // Test erase(const_iterator pos) - single element
  {
    ArrayType a = { 1, 2, 3, 4, 5 };

    // Erase first element
    auto it = a.erase(a.begin());
    ASSERT_EQ(a.size(), 4);
    ASSERT_EQ(a[0], 2);
    ASSERT_EQ(a[1], 3);
    ASSERT_EQ(a[2], 4);
    ASSERT_EQ(a[3], 5);
    ASSERT_EQ(*it, 2); // Returned iterator points to next element

    // Erase middle element
    a = { 1, 2, 3, 4, 5 };
    it = a.erase(a.begin() + 2); // Erase '3'
    ASSERT_EQ(a.size(), 4);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(a[2], 4);
    ASSERT_EQ(a[3], 5);
    ASSERT_EQ(*it, 4);

    // Erase last element
    a = { 1, 2, 3, 4, 5 };
    it = a.erase(a.end() - 1); // Erase '5'
    ASSERT_EQ(a.size(), 4);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(a[2], 3);
    ASSERT_EQ(a[3], 4);
    ASSERT_EQ(it, a.end()); // Returned iterator is end()
  }

  // Test erase(const_iterator first, const_iterator last) - range
  {
    // Erase from beginning
    ArrayType a = { 1, 2, 3, 4, 5 };
    auto it = a.erase(a.begin(), a.begin() + 2); // Erase [1, 2]
    ASSERT_EQ(a.size(), 3);
    ASSERT_EQ(a[0], 3);
    ASSERT_EQ(a[1], 4);
    ASSERT_EQ(a[2], 5);
    ASSERT_EQ(*it, 3);

    // Erase from middle
    a = { 1, 2, 3, 4, 5 };
    it = a.erase(a.begin() + 1, a.begin() + 3); // Erase [2, 3]
    ASSERT_EQ(a.size(), 3);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 4);
    ASSERT_EQ(a[2], 5);
    ASSERT_EQ(*it, 4);

    // Erase to end
    a = { 1, 2, 3, 4, 5 };
    it = a.erase(a.begin() + 2, a.end()); // Erase [3, 4, 5]
    ASSERT_EQ(a.size(), 2);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(it, a.end());

    // Erase full range
    a = { 1, 2, 3, 4, 5 };
    it = a.erase(a.begin(), a.end());
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(it, a.end());

    // Erase empty range (first == last)
    a = { 1, 2, 3, 4, 5 };
    it = a.erase(a.begin() + 2, a.begin() + 2);
    ASSERT_EQ(a.size(), 5);
    ASSERT_EQ(*it, 3); // Points to element at position 2
  }

  // Test with const_iterator (const array)
  {
    const UniqueArray<value_type> ca = { 1, 2, 3, 4, 5 };
    ArrayType a = ca;

    // erase with const_iterator position
    auto it = a.erase(a.cbegin() + 1); // Erase '2'
    ASSERT_EQ(a.size(), 4);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 3);
    ASSERT_EQ(*it, 3);

    // erase with const_iterator range
    a = ca;
    it = a.erase(a.cbegin() + 1, a.cbegin() + 3); // Erase [2, 3]
    ASSERT_EQ(a.size(), 3);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 4);
    ASSERT_EQ(*it, 4);
  }
}

template <typename ArrayType>
void _testArrayPopBack()
{
  using namespace Arccore;

  ArrayType a = {1, 2, 3, 4, 5};
  a.pop_back();
  ASSERT_EQ(a.size(), 4);
  ASSERT_EQ(a[3], 4);

  a.pop_back();
  ASSERT_EQ(a.size(), 3);
  ASSERT_EQ(a[2], 3);

  // Pop until empty
  while (!a.empty())
    a.pop_back();
  ASSERT_EQ(a.size(), 0);
}

template <typename ArrayType>
void _testArrayEmplaceBack()
{
  using namespace Arccore;
  using value_type = ArrayType::value_type;
  // Test with simple type
  if constexpr(std::is_same_v<Int32,value_type>){
    ArrayType a;
    a.emplace_back(42);
    ASSERT_EQ(a.size(), 1);
    ASSERT_EQ(a[0], 42);

    a.emplace_back(100);
    ASSERT_EQ(a.size(), 2);
    ASSERT_EQ(a[1], 100);
  }

  // Test with non-POD type (IntSubClass)
  if constexpr(std::is_same_v<IntSubClass,value_type>){
    ArrayType a;
    a.emplace_back(IntSubClass(10));
    ASSERT_EQ(a.size(), 1);
    ASSERT_EQ(a[0].value(), 10);

    a.emplace_back(IntSubClass(20));
    ASSERT_EQ(a.size(), 2);
    ASSERT_EQ(a[1].value(), 20);
  }
}

template <typename ArrayType>
void _testArrayEdgeCases()
{
  using namespace Arccore;
  using value_type = ArrayType::value_type;

  // Test on empty array
  {
    ArrayType a;
    ASSERT_EQ(a.size(), 0);
    // assign on empty
    a.assign(3l, value_type(7));
    ASSERT_EQ(a.size(), 3);
    // pop_back on empty should not crash (but is undefined behavior in STL)
    // We don't test pop_back on empty as it's undefined
  }

  // Test on single element array
  {
    ArrayType a = { 42 };
    ASSERT_EQ(a.size(), 1);

    // erase single element
    auto it = a.erase(a.begin());
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(it, a.end());

    // assign on single element
    a.assign(1l, value_type(99));
    ASSERT_EQ(a.size(), 1);
    ASSERT_EQ(a[0], 99);

    // pop_back on single element
    a.pop_back();
    ASSERT_EQ(a.size(), 0);

    // emplace_back on single element
    a.emplace_back(123);
    ASSERT_EQ(a.size(), 1);
    ASSERT_EQ(a[0], 123);
  }

  // Test assign with self-assignment-like scenario
  {
    ArrayType a = { 1, 2, 3 };
    auto begin = a.begin();
    auto end = a.end();
    a.assign(begin, end); // Should work but be a no-op effectively
    ASSERT_EQ(a.size(), 3);
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(a[2], 3);
  }
}

} // namespace

TEST(Array, Assign)
{
  _testArrayAssign<UniqueArray<Int32>>();
  _testArrayAssign<SharedArray<Int32>>();
  _testArrayAssign<UniqueArray<IntSubClass>>();
  _testArrayAssign<SharedArray<IntSubClass>>();
}

TEST(Array, Erase)
{
  _testArrayErase<UniqueArray<Int32>>();
  _testArrayErase<SharedArray<Int32>>();
  _testArrayErase<UniqueArray<IntSubClass>>();
  _testArrayErase<SharedArray<IntSubClass>>();
}

TEST(Array, PopBack)
{
  _testArrayPopBack<UniqueArray<Int32>>();
  _testArrayPopBack<SharedArray<Int32>>();
  _testArrayPopBack<UniqueArray<IntSubClass>>();
  _testArrayPopBack<SharedArray<IntSubClass>>();
}

TEST(Array, EmplaceBack)
{
  _testArrayEmplaceBack<UniqueArray<Int32>>();
  _testArrayEmplaceBack<SharedArray<Int32>>();
  _testArrayEmplaceBack<UniqueArray<IntSubClass>>();
  _testArrayEmplaceBack<SharedArray<IntSubClass>>();
}

TEST(Array, EdgeCases)
{
  _testArrayEdgeCases<UniqueArray<Int32>>();
  _testArrayEdgeCases<SharedArray<Int32>>();
  _testArrayEdgeCases<UniqueArray<IntSubClass>>();
  _testArrayEdgeCases<SharedArray<IntSubClass>>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer IntPtrSubClass::count = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Container, typename SubClass>
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
      ARCCORE_UT_CHECK((c.size() == 3), "Bad size (3)");
      ARCCORE_UT_CHECK((c[0] == 1), "Bad value [0]");
      ARCCORE_UT_CHECK((c[1] == 2), "Bad value [1]");
      ARCCORE_UT_CHECK((c[2] == 3), "Bad value [2]");
      c.resize(0);
      ARCCORE_UT_CHECK((c.size() == 0), "Bad size (0)");
      c.resize(5);
      ARCCORE_UT_CHECK((c.size() == 5), "Bad size");
      c.add(SubClass(6));
      ARCCORE_UT_CHECK((c.size() == 6), "Bad size");
      ARCCORE_UT_CHECK((c[5] == 6), "Bad value [5]");
      c.shrink();
      ASSERT_EQ(c.size(), c.capacity()) << "Bad capacity (test 1)";
      c.resize(12);
      c.shrink();
      ASSERT_EQ(c.size(), c.capacity()) << "Bad capacity (test 2)";
    }
    {
      Container c;
      c.shrink();
      ASSERT_EQ(c.capacity(), 0) << "Bad capacity (test 3)";
    }
    {
      Container c;
      Integer nb = 20;
      for (Integer i = 0; i < nb; ++i)
        c.add(SubClass(i));
      c.reserve(nb * 2);
      Int64 current_capacity = c.capacity();
      ASSERT_EQ(current_capacity, (nb * 2)) << "Bad capacity (test 4)";
      c.shrink(c.capacity() + 5);
      ASSERT_EQ(c.capacity(), current_capacity) << "Bad capacity (test 5)";
      c.shrink(32);
      ASSERT_EQ(c.capacity(), 32) << "Bad capacity (test 6)";
      c.shrink();
      ASSERT_EQ(c.capacity(), c.size()) << "Bad capacity (test 7)";
    }
    {
      UniqueArray<Container> uc;
      Integer nb = 1000000;
      uc.resize(1000);
      for (Container& c : uc) {
        c.reserve(nb);
        c.shrink_to_fit();
      }
    }
    {
      Container c;
      for (Integer i = 0; i < 50; ++i)
        c.add(SubClass(i + 2));
      {
        Container c2 = c;
        for (Integer i = 50; i < 100; ++i) {
          c2.add(SubClass(i + 2));
        }
        Container c4 = c2;
        Container c3;
        c3.add(5);
        c3.add(6);
        c3.add(7);
        c3 = c2;
        for (Integer i = 100; i < 150; ++i) {
          c4.add(SubClass(i + 2));
        }
      }
      ARCCORE_UT_CHECK((c.size() == 150), "Bad size (150)");
      c.reserve(300);
      ARCCORE_UT_CHECK((c.capacity() == 300), "Bad capacity (300)");
      for (Integer i = 0; i < 50; ++i) {
        c.remove(i);
      }
      ARCCORE_UT_CHECK((c.size() == 100), "Bad size (100)");
      for (Integer i = 0; i < 50; ++i) {
        //cout << "** VAL: " << i << " c=" << c[i] << " expected=" << ((i*2)+3) << '\n';
        ARCCORE_UT_CHECK((c[i] == ((i * 2) + 3)), "Bad value");
      }
      for (Integer i = 50; i < 100; ++i) {
        //cout << "** VAL: " << i << " c=" << c[i] << " expected=" << (i+52) << '\n';
        ARCCORE_UT_CHECK((c[i] == (i + 52)), "Bad value");
      }
    }
  }
};

namespace
{
void _testArrayNewInternal()
{
  using namespace Arccore;
  std::cout << "** TEST VECTOR NEW\n";

  std::cout << "** wanted_size = " << AlignedMemoryAllocator3::simdAlignment() << "\n";
  //if (impl_size!=wanted_size)
  //ARCCORE_FATAL("Bad sizeof(ArrayImplBase) v={0} expected={1}",impl_size,wanted_size);
  {
    IntegerArrayTester<SharedArray<IntSubClass>, IntSubClass> rvt;
    rvt.test();
  }
  std::cout << "** TEST VECTOR NEW 2\n";
  {
    IntegerArrayTester<SharedArray<IntPtrSubClass>, IntPtrSubClass> rvt;
    rvt.test();
    std::cout << "** COUNT = " << IntPtrSubClass::count << "\n";
  }
  std::cout << "** TEST VECTOR NEW 3\n";
  {
    IntegerArrayTester<SharedArray<IntPtrSubClass>, IntPtrSubClass> rvt;
    rvt.test();
    std::cout << "** COUNT = " << IntPtrSubClass::count << "\n";
  }
  {
    SharedArray<IntSubClass> c;
    c.add(5);
    c.add(7);
    SharedArray<IntSubClass> c2(c.clone());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size() == 3), "Bad value [3]");
    ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
    ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c2[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[2] == 3), "Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c;
    c.add(5);
    c.add(7);
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size() == 3), "Bad value [3]");
    ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
    ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c2[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[2] == 3), "Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c{ 5, 7 };
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size() == 3), "Bad value [3]");
    ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
    ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c2[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[2] == 3), "Bad value [7]");
  }
  {
    UniqueArray<IntSubClass> c{ 5, 7 };
    ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
    ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
  }
  {
    SharedArray<IntSubClass> c{ 5, 7 };
    ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
    ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
  }
  {
    PrintableMemoryAllocator allocator;
    UniqueArray<IntSubClass> c(&allocator);
    UniqueArray<IntSubClass> cx(&allocator, 5);
    ARCCORE_UT_CHECK((cx.size() == 5), "Bad value [5]");
    c.add(5);
    c.add(7);
    UniqueArray<IntSubClass> c2(c.constView());
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size() == 3), "Bad value [3]");
    ARCCORE_UT_CHECK((c.size() == 2), "Bad value [2]");
    ARCCORE_UT_CHECK((c[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c2[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[2] == 3), "Bad value [7]");
    for (Integer i = 0; i < 50; ++i)
      c.add(i + 3);
    c.resize(24);
    c.reserve(70);
  }
  {
    SharedArray<IntSubClass> c2;
    c2.add(3);
    c2.add(5);
    SharedArray<IntSubClass> c22(33);
    ASSERT_EQ(c22.size(), 33);
    c22 = c2;
    {
      SharedArray<IntSubClass> c;
      c.add(5);
      c.add(7);
      c2 = c;
      SharedArray<IntSubClass> c3(c2);
      SharedArray<IntSubClass> c4(c);
      c2.resize(125);
      ASSERT_EQ(c2.size(), c3.size());
      ASSERT_EQ(c2.size(), c4.size());
      c3.resize(459);
      ASSERT_EQ(c2.size(), c3.size());
      ASSERT_EQ(c2.size(), c4.size());
      c4.resize(932);
      c.resize(32);
    }
    c2.add(3);
    ARCCORE_UT_CHECK((c2.size() == 33), "Bad value [3]");
    ARCCORE_UT_CHECK((c2[0] == 5), "Bad value [5]");
    ARCCORE_UT_CHECK((c2[1] == 7), "Bad value [7]");
    ARCCORE_UT_CHECK((c2[32] == 3), "Bad value [7]");
    c2.resize(1293);
    ASSERT_EQ(c2.size(), 1293);
    ASSERT_EQ(c22.size(), 2);

    {
      SharedArray<IntSubClass> values1 = { -7, 3, 4 };
      ASSERT_EQ(values1.size(), 3);
      ASSERT_EQ(values1[0], -7);
      ASSERT_EQ(values1[1], 3);
      ASSERT_EQ(values1[2], 4);
      values1 = { 2, -1, 9, 13 };
      ASSERT_EQ(values1.size(), 4);
      ASSERT_EQ(values1[0], 2);
      ASSERT_EQ(values1[1], -1);
      ASSERT_EQ(values1[2], 9);
      ASSERT_EQ(values1[3], 13);
      SharedArray<IntSubClass> values2 = values1;
      ASSERT_EQ(values2, values1);

      values1 = {};
      ASSERT_EQ(values1.size(), 0);
      ASSERT_EQ(values2.size(), 0);
    }
  }
  {
    UniqueArray<Int32> values1 = { 2, 5 };
    UniqueArray<Int32> values2 = { 4, 9, 7 };
    // Copy the values of values2 to the end of values1.
    std::copy(std::begin(values2), std::end(values2), std::back_inserter(values1));
    std::cout << "** VALUES1 = " << values1 << "\n";
    ARCCORE_UT_CHECK((values1.size() == 5), "BI: Bad size");
    ARCCORE_UT_CHECK((values1[0] == 2), "BI: Bad value [0]");
    ARCCORE_UT_CHECK((values1[1] == 5), "BI: Bad value [1]");
    ARCCORE_UT_CHECK((values1[2] == 4), "BI: Bad value [2]");
    ARCCORE_UT_CHECK((values1[3] == 9), "BI: Bad value [3]");
    ARCCORE_UT_CHECK((values1[4] == 7), "BI: Bad value [4]");

    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << '\n';

    values1 = { -7, 3 };
    ASSERT_EQ(values1.size(), 2);
    ASSERT_EQ(values1[0], -7);
    ASSERT_EQ(values1[1], 3);

    values1 = {};
    ASSERT_EQ(values1.size(), 0);
  }
  {
    UniqueArray<Int32> values1;
    UniqueArray<Int32> values2 = { 4, 9, 7, 6, 3 };
    // Copy the values of values2 to the end of values1.
    values1.copy(values2);
    std::cout << "** VALUES1 = " << values1 << "\n";
    ARCCORE_UT_CHECK((values1.size() == 5), "BI: Bad size");
    ARCCORE_UT_CHECK((values1[0] == 4), "BI2: Bad value [0]");
    ARCCORE_UT_CHECK((values1[1] == 9), "BI2: Bad value [1]");
    ARCCORE_UT_CHECK((values1[2] == 7), "BI2: Bad value [2]");
    ARCCORE_UT_CHECK((values1[3] == 6), "BI2: Bad value [3]");
    ARCCORE_UT_CHECK((values1[4] == 3), "BI2: Bad value [4]");

    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << '\n';
  }
  {
    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
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
    UniqueArray<IntPtrSubClass>::iterator i = std::begin(vx);
    UniqueArray<IntPtrSubClass>::iterator iend = std::end(vx);
    UniqueArray<IntPtrSubClass>::const_iterator ci = i;
    std::cout << "V=" << i->m_v << " " << ci->m_v << " " << (iend - i) << '\n';
    const UniqueArray<IntPtrSubClass>& cvx = (const UniqueArray<IntPtrSubClass>&)(vx);
    UniqueArray<IntPtrSubClass>::const_iterator cicvx = std::begin(cvx);
    std::cout << "V=" << cicvx->m_v << '\n';
    std::copy(std::begin(vx), std::end(vx), std::begin(vx));
  }
  {
    UniqueArray<Int32> values = { 4, 9, 7 };
    for (typename ArrayView<Int32>::const_iter i(values.view()); i(); ++i) {
      std::cout << *i << '\n';
    }
    for (typename ConstArrayView<Int32>::const_iter i(values.view()); i(); ++i) {
      std::cout << *i << '\n';
    }
    for (auto i : values.range()) {
      std::cout << i << '\n';
    }
    for (auto i : values.constView().range()) {
      std::cout << i << '\n';
    }

    {
      auto r1 = std::make_reverse_iterator(values.end());
      auto r2 = std::make_reverse_iterator(values.begin());
      for (; r1 != r2; ++r1) {
        std::cout << "RVALUE = " << *r1 << '\n';
      }
    }
    {
      auto r1 = values.rbegin();
      ASSERT_EQ((*r1), 7);
      ++r1;
      ASSERT_EQ((*r1), 9);
      ++r1;
      ASSERT_EQ((*r1), 4);
      ++r1;
      ASSERT_TRUE((r1 == values.rend()));
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
      SmallSpan<const IntSubClassNoPod> c2_small_span = c2;
      ASSERT_EQ(c.constSpan(), c2.constSpan());
      ASSERT_EQ(c.constSmallSpan(), c2_small_span);
      ASSERT_EQ(c.smallSpan(), c2.smallSpan());
    }
  }
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc)
{
  try {
    _testArrayNewInternal();
  }
  catch (const Exception& ex) {
    std::cerr << "Exception ex=" << ex << "\n";
    throw;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void _Add(Array<Real>& v, Integer new_size)
{
  v.resize(new_size);
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, Misc2)
{
  using namespace Arccore;
  {
    UniqueArray<Real> v;
    v.resize(3);
    v[0] = 1.2;
    v.at(1) = -1.3;
    v[2] = 7.6;
    ASSERT_EQ(v[0], 1.2);
    ASSERT_EQ(v[1], -1.3);
    ASSERT_EQ(v.at(2), 7.6);

    v.add(4.3);
    for (Real x : v) {
      std::cout << " Value: " << x << '\n';
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

TEST(Array, SubViews)
{
  using namespace Arccore;

  {
    // Test Array::subView() and Array::subConstView()
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

class NoCopyData
{
 public:

  NoCopyData(const NoCopyData& x) = delete;
};

template <typename DataType>
class MyArrayTest
: public Arccore::Array<DataType>
{
 public:

  using BaseClass = Arccore::Array<DataType>;
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

    c.resize(9, IntSubClassNoPod(ref_value1));
    std::cout << "C1=" << c << "\n";
    for (IntSubClassNoPod x : c)
      ASSERT_EQ(x, ref_value1);

    c.resize(21, IntSubClassNoPod(ref_value2));
    ASSERT_EQ(c.size(), 21);
    std::cout << "C2=" << c << "\n";

    // Resize without initializing. The values for elements
    // from 9 to 18 must be \a ref_value2
    c.resizeNoInit(18);
    std::cout << "C4=" << c << "\n";
    for (Int32 i = 9, s = c.size(); i < s; ++i)
      ASSERT_EQ(c[i], ref_value2);
    for (Int32 i = 9, s = c.size(); i < s; ++i)
      new (c.data() + i) IntSubClassNoPod(i + 2);
    for (Int32 i = 9, s = c.size(); i < s; ++i)
      ASSERT_EQ(c[i], (i + 2));
  }
  {
    MyArrayTest<NoCopyData> c2;
    c2.resizeNoInit(25);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ArrayType>
class AllocatorTest
{
 public:

  static void doTestBase()
  {
    using namespace Arccore;

    IMemoryAllocator* allocator1 = AlignedMemoryAllocator::Simd();
    PrintableMemoryAllocator printable_allocator2;
    IMemoryAllocator* allocator2 = &printable_allocator2;

    {
      std::cout << "Array a\n";
      ArrayType a(allocator1);
      ASSERT_EQ(a.allocator(), allocator1);
      a.add(27);
      a.add(38);
      a.add(13);
      a.add(-5);

      std::cout << "Array b\n";
      ArrayType b(allocator2);
      ASSERT_EQ(b.capacity(), 0);
      ASSERT_EQ(b.size(), 0);
      ASSERT_EQ(b.allocator(), allocator2);

      b = a;
      ASSERT_EQ(b.size(), a.size());
      ASSERT_EQ(b.allocator(), a.allocator());

      std::cout << "Array c\n";
      ArrayType c(a.clone());
      ASSERT_EQ(c.allocator(), a.allocator());
      ASSERT_EQ(c.size(), a.size());
      ASSERT_EQ(c.constSpan(), a.constSpan());

      std::cout << "Array d\n";
      ArrayType d(allocator2, a);
      ASSERT_EQ(d.allocator(), allocator2);
      ASSERT_EQ(d.size(), a.size());
      ASSERT_EQ(d.constSpan(), a.constSpan());

      std::cout << "Array e\n";
      ArrayType e(allocator2, 25);
      ASSERT_EQ(e.allocator(), allocator2);
      ASSERT_EQ(e.size(), 25);

      ArrayType f(allocator2);
      f = e;
      ASSERT_EQ(f.allocator(), e.allocator());
      ASSERT_EQ(f.size(), e.size());

      f = f;
      ASSERT_EQ(f.allocator(), e.allocator());
      ASSERT_EQ(f.size(), e.size());
    }
  }
};

TEST(UniqueArray, AllocatorBase)
{
  AllocatorTest<UniqueArray<Int32>>::doTestBase();
}

TEST(SharedArray, AllocatorBase)
{
  AllocatorTest<SharedArray<Int32>>::doTestBase();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(UniqueArray, Allocator)
{
  using namespace Arccore;
  std::cout << "Sizeof(MemoryAllocationOptions)=" << sizeof(MemoryAllocationOptions) << "\n";
  std::cout << "Sizeof(ArrayMetaData)=" << sizeof(ArrayMetaData) << "\n";
  std::cout << "Sizeof(UniqueArray<Int32>)=" << sizeof(UniqueArray<Int32>) << "\n";
  std::cout << "Sizeof(SharedArray<Int32>)=" << sizeof(SharedArray<Int32>) << "\n";

  PrintableMemoryAllocator printable_allocator;
  PrintableMemoryAllocator printable_allocator2;
  IMemoryAllocator* allocator1 = AlignedMemoryAllocator3::Simd();
  IMemoryAllocator* allocator2 = AlignedMemoryAllocator::Simd();
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
    for (Integer i = 0; i < 2; ++i) {
      array[i] = UniqueArray<Int32>(allocator3);
    }
    ASSERT_EQ(array[0].allocator(), allocator3);
    ASSERT_EQ(array[1].allocator(), allocator3);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(SharedArray, Allocator)
{
  using namespace Arccore;
  std::cout << "Sizeof(MemoryAllocationOptions)=" << sizeof(MemoryAllocationOptions) << "\n";
  std::cout << "Sizeof(ArrayMetaData)=" << sizeof(ArrayMetaData) << "\n";
  std::cout << "Sizeof(UniqueArray<Int32>)=" << sizeof(UniqueArray<Int32>) << "\n";
  std::cout << "Sizeof(SharedArray<Int32>)=" << sizeof(SharedArray<Int32>) << "\n";

  PrintableMemoryAllocator printable_allocator;
  PrintableMemoryAllocator printable_allocator2;
  IMemoryAllocator* allocator1 = AlignedMemoryAllocator3::Simd();
  IMemoryAllocator* allocator2 = AlignedMemoryAllocator::Simd();
  {
    std::cout << "Array a1\n";
    SharedArray<Int32> a1(allocator2);
    ASSERT_EQ(a1.allocator(), allocator2);
    ASSERT_EQ(a1.size(), 0);
    ASSERT_EQ(a1.capacity(), 0);
    ASSERT_EQ(a1.data(), nullptr);

    std::cout << "Array a2\n";
    SharedArray<Int32> a2(a1);
    ASSERT_EQ(a1.allocator(), a2.allocator());
    ASSERT_EQ(a2.capacity(), 0);
    ASSERT_EQ(a2.data(), nullptr);
    a1.reserve(3);
    a1.add(5);
    a1.add(7);
    a1.add(12);
    a1.add(3);
    a1.add(1);
    ASSERT_EQ(a1.size(), 5);

    std::cout << "Array a3\n";
    SharedArray<Int32> a3(MemoryAllocationOptions{ allocator1 });
    a3.add(4);
    a3.add(6);
    a3.add(2);
    ASSERT_EQ(a3.size(), 3);
    a3 = a1;
    ASSERT_EQ(a3.allocator(), a1.allocator());
    ASSERT_EQ(a3.size(), a1.size());
    ASSERT_EQ(a3.constSpan(), a1.constSpan());

    std::cout << "Array a4\n";
    SharedArray<Int32> a4(allocator1, 2);
    ASSERT_EQ(a4.size(), 2);
    a4.add(4);
    a4.add(6);
    a4.add(2);
    ASSERT_EQ(a4.size(), 5);
    ASSERT_EQ(a4[2], 4);
    ASSERT_EQ(a4[3], 6);
    ASSERT_EQ(a4[4], 2);

    a4 = a1.span();
    ASSERT_EQ(a4.allocator(), allocator1);

    a4 = SharedArray<Int32>(&printable_allocator2);

    SharedArray<Int32> array[2];
    IMemoryAllocator* allocator3 = allocator1;
    for (Integer i = 0; i < 2; ++i) {
      array[i] = SharedArray<Int32>(allocator3);
    }
    ASSERT_EQ(array[0].allocator(), allocator3);
    ASSERT_EQ(array[1].allocator(), allocator3);
  }
}

/*!
 * \brief Allocator for testing arguments.
 *
 * Allows verification that it was called with the correct arguments.
 * It should only be called with args.memoryLocationHint() which equals
 * eMemoryLocationHint::None or eMemoryLocationHint::HostAndDeviceMostlyRead
 */
class TesterMemoryAllocatorV3
: public IMemoryAllocator3
{
 public:

  bool hasRealloc(MemoryAllocationArgs args) const override
  {
    _checkValid(args);
    return m_default_allocator.hasRealloc(args);
  }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override
  {
    _checkValid(args);
    return m_default_allocator.allocate(args, new_size);
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override
  {
    _checkValid(args);
    return m_default_allocator.reallocate(args, current_ptr, new_size);
  }
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override
  {
    _checkValid(args);
    m_default_allocator.deallocate(args, ptr);
  }
  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const override
  {
    _checkValid(args);
    return m_default_allocator.adjustedCapacity(args, wanted_capacity, element_size);
  }
  size_t guaranteedAlignment(MemoryAllocationArgs args) const override
  {
    _checkValid(args);
    return m_default_allocator.guaranteedAlignment(args);
  }

  void notifyMemoryArgsChanged(MemoryAllocationArgs old_args, MemoryAllocationArgs new_args, AllocatedMemoryInfo ptr) override
  {
    // This method is only called once so we test the expected values directly
    ASSERT_EQ(old_args.memoryLocationHint(), eMemoryLocationHint::None);
    ASSERT_EQ(new_args.memoryLocationHint(), eMemoryLocationHint::MainlyHost);
    ASSERT_EQ(ptr.size(), 8);
    m_default_allocator.notifyMemoryArgsChanged(old_args, new_args, ptr);
  }

 private:

  DefaultMemoryAllocator3 m_default_allocator;

 private:

  static void _checkValid(MemoryAllocationArgs args)
  {
    bool v1 = args.memoryLocationHint() == eMemoryLocationHint::None;
    bool v2 = args.memoryLocationHint() == eMemoryLocationHint::MainlyHost;
    bool v3 = args.memoryLocationHint() == eMemoryLocationHint::HostAndDeviceMostlyRead;
    bool is_valid = v1 || v2 || v3;
    ASSERT_TRUE(is_valid);
  }
};

#define ASSERT_SAME_ARRAY_INFOS(a, b) \
  ASSERT_EQ(a.allocationOptions(), b.allocationOptions()); \
  ASSERT_EQ(a.size(), b.size()); \
  ASSERT_EQ(a.capacity(), b.capacity())

TEST(Array, AllocatorV2)
{
  using namespace Arccore;
  TesterMemoryAllocatorV3 testerv2_allocator;
  TesterMemoryAllocatorV3 testerv2_allocator2;
  MemoryAllocationOptions allocate_options1(&testerv2_allocator, eMemoryLocationHint::HostAndDeviceMostlyRead);
  MemoryAllocationOptions allocate_options2(&testerv2_allocator, eMemoryLocationHint::None, 0);
  {
    MemoryAllocationOptions opt3(&testerv2_allocator);
    opt3.setMemoryLocationHint(eMemoryLocationHint::HostAndDeviceMostlyRead);
    ASSERT_EQ(opt3, allocate_options1);
  }
  {
    std::cout << "Array a1\n";
    UniqueArray<Int32> a1(allocate_options2);
    ASSERT_EQ(a1.allocationOptions(), allocate_options2);
    ASSERT_EQ(a1.size(), 0);
    ASSERT_EQ(a1.capacity(), 0);
    ASSERT_EQ(a1.data(), nullptr);

    std::cout << "Array a2\n";
    UniqueArray<Int32> a2(a1);
    ASSERT_SAME_ARRAY_INFOS(a2, a1);
    ASSERT_EQ(a2.data(), nullptr);
    a1.reserve(3);
    a1.add(5);
    a1.add(7);
    a1.add(12);
    a1.add(3);
    a1.add(1);
    ASSERT_EQ(a1.size(), 5);
    a2.add(9);
    a2.add(17);
    // To test notifyMemoryArgsChanged()
    a2.setMemoryLocationHint(eMemoryLocationHint::MainlyHost);

    std::cout << "Array a3\n";
    UniqueArray<Int32> a3(allocate_options1);
    a3.add(4);
    a3.add(6);
    a3.add(2);
    ASSERT_EQ(a3.size(), 3);
    a3 = a1;
    ASSERT_EQ(a3.allocator(), a1.allocator());
    ASSERT_EQ(a3.size(), a1.size());
    ASSERT_EQ(a3.constSpan(), a1.constSpan());

    std::cout << "Array a4\n";
    UniqueArray<Int32> a4(allocate_options1);
    a4.add(4);
    a4.add(6);
    a4.add(2);
    ASSERT_EQ(a4.size(), 3);
    a4 = a1.span();
    ASSERT_EQ(a4.allocationOptions(), allocate_options1);

    a4 = UniqueArray<Int32>(&testerv2_allocator2);

    UniqueArray<Int32> array[2];
    MemoryAllocationOptions allocator3 = allocate_options1;
    for (Integer i = 0; i < 2; ++i) {
      array[i] = UniqueArray<Int32>(allocator3);
    }
    ASSERT_EQ(array[0].allocationOptions(), allocator3);
    ASSERT_EQ(array[1].allocationOptions(), allocator3);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array, DebugInfo)
{
  using namespace Arccore;
  DefaultMemoryAllocator3 m_default_allocator;
  MemoryAllocationOptions allocate_options2(&m_default_allocator, eMemoryLocationHint::None, 0);

  String a1_name("Array1");
  String sa1_name("SharedArray1");
  UniqueArray<Int32> a3;
  {
    std::cout << "Array a1\n";
    UniqueArray<Int32> a1(allocate_options2);
    a1.setDebugName(a1_name);
    ASSERT_EQ(a1.allocationOptions(), allocate_options2);
    ASSERT_EQ(a1.size(), 0);
    ASSERT_EQ(a1.capacity(), 0);
    ASSERT_EQ(a1.data(), nullptr);
    ASSERT_EQ(a1.debugName(), a1_name);

    std::cout << "SharedArray sa1\n";
    SharedArray<Int32> sa1(allocate_options2);
    sa1.setDebugName(sa1_name);
    ASSERT_EQ(sa1.allocationOptions(), allocate_options2);
    ASSERT_EQ(sa1.size(), 0);
    ASSERT_EQ(sa1.capacity(), 0);
    ASSERT_EQ(sa1.data(), nullptr);
    ASSERT_EQ(sa1.debugName(), sa1_name);

    ASSERT_EQ(a1.debugName(), a1_name);

    std::cout << "Array a2\n";
    UniqueArray<Int32> a2(a1);
    ASSERT_SAME_ARRAY_INFOS(a2, a1);
    ASSERT_EQ(a2.data(), nullptr);
    ASSERT_EQ(a2.debugName(), a1_name);
    a1.reserve(3);
    a1.add(5);
    a1.add(7);
    a1.add(12);
    a1.add(3);
    a1.add(1);
    ASSERT_EQ(a1.size(), 5);
    a2.add(9);
    a2.add(17);
    ASSERT_EQ(a2.debugName(), a1_name);
    ASSERT_EQ(a2.size(), 2);

    a3 = a2;
  }
  ASSERT_EQ(a3.debugName(), a1_name);
  ASSERT_EQ(a3.size(), 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Collections, Memory)
{
  using namespace Arccore;
  std::cout << eHostDeviceMemoryLocation::Unknown << " "
            << eHostDeviceMemoryLocation::Device << " "
            << eHostDeviceMemoryLocation::Host << " "
            << eHostDeviceMemoryLocation::ManagedMemoryDevice << " "
            << eHostDeviceMemoryLocation::ManagedMemoryHost << "\n";

  std::cout << eMemoryResource::Unknown << " "
            << eMemoryResource::Host << " "
            << eMemoryResource::HostPinned << " "
            << eMemoryResource::Device << " "
            << eMemoryResource::UnifiedMemory << "\n";

  UniqueArray<Int32> a1;
  ASSERT_EQ(a1.hostDeviceMemoryLocation(), eHostDeviceMemoryLocation::Unknown);
  a1._internalSetHostDeviceMemoryLocation(eHostDeviceMemoryLocation::Host);
  ASSERT_EQ(a1.hostDeviceMemoryLocation(), eHostDeviceMemoryLocation::Host);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// Explicitly instantiates array classes to ensure
// that all methods work
template class UniqueArray<IntSubClass>;
template class SharedArray<IntSubClass>;
template class Array<IntSubClass>;
template class AbstractArray<IntSubClass>;
} // namespace Arcane
