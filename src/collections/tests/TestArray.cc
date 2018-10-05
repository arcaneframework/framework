#include <gtest/gtest.h>

#include "arccore/collections/Array.h"

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
    UniqueArray<IntPtrSubClass> vx;
    vx.add(IntPtrSubClass(5));
    auto range = vx.range();
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
    auto r = vx.range();
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
  }
}
}

TEST(Array, Misc)
{
  _testArrayNewInternal();
}
