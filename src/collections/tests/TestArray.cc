#include <gtest/gtest.h>

#include "arccore/collections/Array.h"

using namespace Arccore;

TEST(Array, Misc)
{
  _testArrayNewInternal();
}

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
};
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
