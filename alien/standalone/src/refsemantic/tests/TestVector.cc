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

#include <alien/ref/AlienRefSemantic.h>

#include <Environment.h>

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
  const Alien::VectorDistribution d(s, AlienTest::Environment::parallelMng());
  const Alien::Vector v(d);
  ASSERT_TRUE(s == v.space());
  ASSERT_EQ(10, v.space().size());
  ASSERT_EQ("MySpace", v.space().name());
}

// Tests the anonymous c'tor.
TEST(TestVector, AnonymousConstructor)
{
  const Alien::VectorDistribution d(10, AlienTest::Environment::parallelMng());
  const Alien::Vector v(d);
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
}

// Tests the anonymous c'tor.
TEST(TestVector, AnonymousConstructor2)
{
  const Alien::Vector v(10, AlienTest::Environment::parallelMng());
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
}

// Tests the rvalue c'tor.
TEST(TestVector, RValueConstructor)
{
  const Alien::Vector v(Alien::Vector(10, AlienTest::Environment::parallelMng()));
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
}

// Tests the rvalue affectation.
TEST(TestVector, RValueAffectation)
{
  Alien::Vector v = Alien::Vector(10, AlienTest::Environment::parallelMng());
  ASSERT_TRUE(Alien::Space(10) == v.space());
  ASSERT_EQ(10, v.space().size());
  v = Alien::Vector(5, AlienTest::Environment::parallelMng());
  ASSERT_TRUE(Alien::Space(5) == v.space());
  ASSERT_EQ(5, v.space().size());
}
