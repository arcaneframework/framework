/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <arccore/message_passing/Messages.h>

#include <Environment.h>

#include <alien/core/utils/Partition.h>
#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>

using namespace Arccore;

// Tests indices per label in Space
TEST(TestSpacePartition, IndicesParLabel)
{
  Alien::Space s(10);
  UniqueArray<Integer> indices1(3);
  indices1[0] = 0;
  indices1[1] = 1;
  indices1[2] = 2;
  s.setField("Field1", indices1);
  UniqueArray<Integer> indices2(3);
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
  UniqueArray<String> tags(2);
  tags[0] = "Field1";
  tags[1] = "Field2";
  UniqueArray<Integer> indices1(3);
  indices1[0] = 0;
  indices1[1] = 1;
  indices1[2] = 2;
  s.setField(tags[0], indices1);
  UniqueArray<Integer> indices2(3);
  indices2[0] = 6;
  indices2[1] = 7;
  indices2[2] = 8;
  s.setField(tags[1], indices2);
  Alien::MatrixDistribution d(10, 10, AlienTest::Environment::parallelMng());
  Alien::Partition p(s, d);
  p.create(tags);
  ASSERT_TRUE(p.hasUntaggedPart());
  ASSERT_EQ(2, p.nbTaggedParts());
  const auto& i1 = p.taggedPart(0);
  for (auto i = 0; i < 3; ++i)
    ASSERT_EQ(indices1[i], i1[i]);
  const auto& i2 = p.taggedPart(1);
  for (auto i = 0; i < 3; ++i)
    ASSERT_EQ(indices2[i], i2[i]);
  ASSERT_EQ(p.tag(0), tags[0]);
  ASSERT_EQ(p.tag(1), tags[1]);
  const auto& i3 = p.untaggedPart();
  auto* pm = AlienTest::Environment::parallelMng();
  int size = i3.size();
  int allSize =
  Arccore::MessagePassing::mpAllReduce(pm, Arccore::MessagePassing::ReduceSum, size);
  UniqueArray<Integer> globali3(allSize, -1);
  ASSERT_EQ(static_cast<Integer>(allSize), 4);
  Arccore::MessagePassing::mpAllGather(pm, i3.constView(), globali3.view());
  ASSERT_EQ(globali3[0], 3);
  ASSERT_EQ(globali3[1], 4);
  ASSERT_EQ(globali3[2], 5);
  ASSERT_EQ(globali3[3], 9);
}
