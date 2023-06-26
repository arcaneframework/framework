#include <gtest/gtest.h>

#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

// Tests the default c'tor.
TEST(TestVector, DefaultConstructor)
{
  const Alien::Vector v;
  ASSERT_EQ(0, v.space().size());
}

// Tests the space c'tor.
TEST(TestVector, SpaceConstructor)
{
  const Alien::Space s(10, "MySpace");
  const Alien::VectorDistribution d(s, Environment::parallelMng());
  const Alien::Vector v(d);
  ASSERT_TRUE(s == v.space());
  ASSERT_EQ(10, v.space().size());
  ASSERT_EQ("MySpace", v.space().name());
}

// Tests the anonymous c'tor.
TEST(TestVector, AnonymousConstructor)
{
  const Alien::VectorDistribution d(10, Environment::parallelMng());
  const Alien::Vector v(d);
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
}

// Tests the anonymous c'tor.
TEST(TestVector, AnonymousConstructor2)
{
  const Alien::Vector v(10, Environment::parallelMng());
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
}

// Tests the rvalue c'tor.
TEST(TestVector, RValueConstructor)
{
  const Alien::Vector v(Alien::Vector(10, Environment::parallelMng()));
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
}

// Tests the rvalue affectation.
TEST(TestVector, RValueAffectation)
{
  Alien::Vector v = Alien::Vector(10, Environment::parallelMng());
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
  v = Alien::Vector(5, Environment::parallelMng());
  ASSERT_TRUE(Alien::Space(5) == v.space());
  ASSERT_EQ(5, v.space().size());
}
