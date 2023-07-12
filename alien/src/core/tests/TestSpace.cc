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
