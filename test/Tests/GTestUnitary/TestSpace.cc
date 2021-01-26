
#include <alien/Alien.h>

#include "gtest/gtest.h"

// Tests the default c'tor.
TEST(TestSpace, DefaultConstructor)
{
  const Alien::Space s;
  ASSERT_EQ(0, s.size());
}

// Tests the c'tor with size (anonymous space).
TEST(TestSpace, ConstructorWithSize)
{
  const Alien::Space s(10);
  ASSERT_EQ(10, s.size());
}

// Tests the c'tor with size and label.
TEST(TestSpace, ConstructorWithSizeAndName)
{
  const Alien::Space s(10, "MySpace");
  ASSERT_EQ(10, s.size());
  ASSERT_EQ("MySpace", s.name());
}

// Tests the equality between strong spaces
TEST(TestSpace, StrongSpaceEquality)
{
  const Alien::Space s1(10, "MySpace");
  ASSERT_TRUE(s1 == s1);
  const Alien::Space s2(5, "MySpace");
  ASSERT_FALSE(s1 == s2);
  const Alien::Space s3(5, "OtherSpace");
  ASSERT_FALSE(s3 == s2);
}

// Tests the equality between anonymous spaces
TEST(TestSpace, AnonymousSpaceEquality)
{
  const Alien::Space s1(10);
  ASSERT_TRUE(s1 == s1);
  const Alien::Space s2(5);
  ASSERT_FALSE(s1 == s2);
}

// Tests the equality between anonymous and string spaces
TEST(TestSpace, AnonymousAndStrongSpaceEquality)
{
  const Alien::Space s1(10, "MySpace");
  const Alien::Space s2(10);
  ASSERT_TRUE(s1 == s2);
}

// Tests the rvalue c'tor
TEST(TestSpace, RValueConstructor)
{
  const Alien::Space s1(10, "MySpace");
  auto f = []() -> Alien::Space { return Alien::Space(10, "MySpace"); };
  Alien::Space s2(f()); // Attention, const déclenche de optimisations...
  ASSERT_TRUE(s1 == s2);
  Alien::Space s3;
  s3 = std::move(s2);
  ASSERT_TRUE(s1 == s3);
}
