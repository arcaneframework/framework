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

#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/move/data/MatrixData.h>

#include <Environment.h>

// Tests the default c'tor.
TEST(TestMatrix, DefaultConstructor)
{
  const Alien::Move::MatrixData m;
  ASSERT_EQ(0, m.colSpace().size());
  ASSERT_EQ(0, m.rowSpace().size());
}

// Tests the space c'tor.
TEST(TestMatrix, SpaceConstructor)
{
  const Alien::Space row_space(9, "MyRowSpace");
  const Alien::Space col_space(11, "MyColSpace");
  const Alien::MatrixDistribution d(row_space, col_space, AlienTest::Environment::parallelMng());
  const Alien::Move::MatrixData m(d);
  ASSERT_TRUE(row_space == m.rowSpace());
  ASSERT_TRUE(col_space == m.colSpace());
  ASSERT_EQ(9, m.rowSpace().size());
  ASSERT_EQ(11, m.colSpace().size());
}

// Tests the anonymous c'tor.
TEST(TestMatrix, AnonymousConstructor)
{
  const Alien::MatrixDistribution d(10, 10, AlienTest::Environment::parallelMng());
  const Alien::Move::MatrixData m(d);
  ASSERT_TRUE(Alien::Space(10) == m.rowSpace());
  ASSERT_TRUE(Alien::Space(10) == m.colSpace());
}

// Tests the rvalue c'tor.
TEST(TestMatrix, RValueConstructor)
{
  const Alien::MatrixDistribution d(10, 10, AlienTest::Environment::parallelMng());
  const auto m = Alien::Move::MatrixData(d);
  ASSERT_TRUE(Alien::Space(10) == m.colSpace());
}

// Tests cloning Matrix
TEST(TestMatrix, Clone)
{
  const Alien::MatrixDistribution d(10, 10, AlienTest::Environment::parallelMng());
  auto m = Alien::Move::MatrixData(d);
  auto m2 = m.clone();
  ASSERT_EQ(10, m2.colSpace().size());
  // Moving out m
  auto m3 = std::move(m);
  ASSERT_EQ(10, m2.rowSpace().size());
}

// Tests replacement
TEST(TestMatrix, Replacement)
{
  const Alien::MatrixDistribution d10(10, 10, AlienTest::Environment::parallelMng());
  auto v = Alien::Move::MatrixData(d10);
  const Alien::MatrixDistribution d5(5, 10, AlienTest::Environment::parallelMng());
  v = Alien::Move::MatrixData(d5);
  ASSERT_TRUE(Alien::Space(5) == v.rowSpace());
  ASSERT_TRUE(Alien::Space(10) == v.colSpace());
}

// Tests replacement
TEST(TestMatrix, RValueAffectation)
{
  const Alien::MatrixDistribution d10(10, 10, AlienTest::Environment::parallelMng());
  auto v = Alien::Move::MatrixData(d10);
  const Alien::MatrixDistribution d5(5, 10, AlienTest::Environment::parallelMng());
  auto v2 = Alien::Move::MatrixData(d5);
  v = std::move(v2);
  ASSERT_TRUE(Alien::Space(5) == v.rowSpace());
}
