#include "gtest/gtest.h"

#include <alien/Alien.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

// Tests indices per label in Space
TEST(TestSpacePartition, IndicesParLabel)
{
  Alien::Space s(10);
  Arccore::UniqueArray<Arccore::Integer> indices1(3);
  indices1[0] = 0;
  indices1[1] = 1;
  indices1[2] = 2;
  s.setField("Field1", indices1);
  Arccore::UniqueArray<Arccore::Integer> indices2(3);
  indices2[0] = 6;
  indices2[1] = 7;
  indices2[2] = 8;
  s.setField("Field2", indices2);
  {
    const auto& i1 = s.field("Field1");
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices1[i], i1[i]);
    const auto& i2 = s.field("Field2");
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices2[i], i2[i]);
  }
  {
    ASSERT_EQ(s.fieldLabel(0), "Field1");
    const auto& i1 = s.field(0);
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices1[i], i1[i]);
    ASSERT_EQ(s.fieldLabel(1), "Field2");
    const auto& i2 = s.field(1);
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices2[i], i2[i]);
  }
  const Alien::Space s2 = s;
  ASSERT_TRUE(s == s2);
  {
    const auto& i1 = s2.field("Field1");
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices1[i], i1[i]);
    const auto& i2 = s2.field("Field2");
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices2[i], i2[i]);
  }
  {
    ASSERT_EQ(s2.fieldLabel(0), "Field1");
    const auto& i1 = s2.field(0);
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices1[i], i1[i]);
    ASSERT_EQ(s2.fieldLabel(1), "Field2");
    const auto& i2 = s2.field(1);
    for (auto i = 0; i < 3; ++i)
      ASSERT_EQ(indices2[i], i2[i]);
  }
}

// Tests indices per label in Space
TEST(TestSpacePartiton, Partition)
{
  Alien::Space s(10);
  Arccore::UniqueArray<Arccore::Integer> indices1(3);
  indices1[0] = 0;
  indices1[1] = 1;
  indices1[2] = 2;
  s.setField("Field1", indices1);
  Arccore::UniqueArray<Arccore::Integer> indices2(3);
  indices2[0] = 6;
  indices2[1] = 7;
  indices2[2] = 8;
  s.setField("Field2", indices2);
  Alien::MatrixDistribution d(10, 10, Environment::parallelMng());
  Alien::Partition p(s, d);
  Arccore::UniqueArray<Arccore::String> tags(2);
  tags[0] = "Field1";
  tags[1] = "Field2";
  p.create(tags);
  ASSERT_TRUE(p.hasUntaggedPart());
  ASSERT_EQ(2, p.nbTaggedParts());
  const auto& i1 = p.taggedPart(0);
  for (auto i = 0; i < 3; ++i)
    ASSERT_EQ(indices1[i], i1[i]);
  const auto& i2 = p.taggedPart(1);
  for (auto i = 0; i < 3; ++i)
    ASSERT_EQ(indices2[i], i2[i]);
  ASSERT_EQ(p.tag(0), "Field1");
  ASSERT_EQ(p.tag(1), "Field2");
  const auto& i3 = p.untaggedPart();
  auto* pm = Environment::parallelMng();
  int size = i3.size();
  int allSize =
      Arccore::MessagePassing::mpAllReduce(pm, Arccore::MessagePassing::ReduceSum, size);
  Arccore::UniqueArray<Arccore::Integer> globali3(allSize, -1);
  ASSERT_EQ(static_cast<Arccore::Integer>(allSize), 4);
  Arccore::ConstArrayView<int> i3_constview(i3);
  Arccore::ArrayView<int> globali3_view(globali3);
  Arccore::MessagePassing::mpAllGather(pm, i3.constView(), globali3.view());
  ASSERT_EQ(globali3[0], 3);
  ASSERT_EQ(globali3[1], 4);
  ASSERT_EQ(globali3[2], 5);
  ASSERT_EQ(globali3[3], 9);
}
