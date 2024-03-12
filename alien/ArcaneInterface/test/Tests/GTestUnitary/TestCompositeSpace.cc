#include "gtest/gtest.h"

#include <alien/Alien.h>

TEST(TestCompositeSpace, DefaultConstructor)
{
  const Alien::CompositeKernel::Space s;
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(0, s.subSpaceSize());
}

TEST(TestCompositeSpace, SpaceEquality)
{
  const Alien::CompositeKernel::Space s1;
  const Alien::ISpace& is1 = s1;;
  ASSERT_TRUE(is1 == s1);
  const Alien::CompositeKernel::Space s2;
  ASSERT_TRUE(is1 == s2);
  const Alien::Space s3;
  ASSERT_TRUE(is1 == s3);
  const Alien::Space s4(1);
  ASSERT_FALSE(is1 == s4);
  const Alien::Space s5(0, "Named");
  ASSERT_FALSE(is1 == s5);
}

TEST(TestCompositeSpace, SubSpaceResize)
{
  Alien::CompositeKernel::Space s;
  Alien::ISpace& s_as_ispace = s;
  s.resizeSubSpace(3);
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(3, s.subSpaceSize());
  s[0].reset(new Alien::Space(1));
  ASSERT_EQ(1, s.size());
  s[0].reset(new Alien::Space(3));
  ASSERT_EQ(3, s.size());
  s[1].reset(new Alien::Space(4, "Named"));
  ASSERT_EQ(7, s.size());
  s[2].reset(new Alien::Space(3));
  ASSERT_EQ(10, s.size());
  const Alien::Space s1(10);
  ASSERT_TRUE(s1 == s_as_ispace);
}

TEST(TestCompositeSpace, SubSpaceMultipleResize)
{
  Alien::CompositeKernel::Space s;
  s.resizeSubSpace(1);
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(1, s.subSpaceSize());
  s[0].reset(new Alien::Space(3));
  ASSERT_EQ(3, s.size());
  s.resizeSubSpace(2);
  ASSERT_EQ(0, s.size());
  ASSERT_EQ(2, s.subSpaceSize());
  s[0].reset(new Alien::Space(3));
  ASSERT_EQ(3, s.size());
  s[1].reset(new Alien::Space(4));
  ASSERT_EQ(7, s.size());
}

TEST(TestCompositeSpace, RValueConstructor)
{
  Alien::CompositeKernel::Space s1;
  s1.resizeSubSpace(1);
  s1[0].reset(new Alien::Space(3));
  auto f = []() -> Alien::CompositeKernel::Space {
    Alien::CompositeKernel::Space s;
    s.resizeSubSpace(1);
    s[0].reset(new Alien::Space(3));
    return s;
  };
  const Alien::CompositeKernel::Space s2 =
      f(); // Attention, const déclenche de optimisations...
  ASSERT_TRUE(s1 == s2);
  Alien::CompositeKernel::Space s3;
  s3 = std::move(s2);
  ASSERT_TRUE(s1 == s3);
}
