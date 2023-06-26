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
#include <alien/data/CompositeMatrix.h>

TEST(TestCompositeMatrix, DefaultConstructor)
{
  Alien::CompositeMatrix v;
  ASSERT_EQ(0, v.size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(0, v.rowSpace().size());
  ASSERT_EQ(0, v.colSpace().size());
}

TEST(TestCompositeMatrix, ConstructorWithSize)
{
  Alien::CompositeMatrix v(3);
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(3, v.size());
  ASSERT_EQ(0, v.rowSpace().size());
  ASSERT_EQ(0, v.colSpace().size());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      auto& c = v(i, j);
      // ASSERT_THROW(c.impl(), Alien::FatalErrorException);
      ASSERT_EQ(0, c.rowSpace().size());
      ASSERT_EQ(0, c.colSpace().size());
    }
  }
}

TEST(TestCompositeMatrix, CompositeConstructorsTest)
{
  Alien::MatrixDistribution dist0(4, 4, AlienTest::Environment::parallelMng());
  Alien::MatrixDistribution dist1(5, 5, AlienTest::Environment::parallelMng());
  Alien::CompositeMatrix v(2);
  Alien::CompositeElement(v, 0, 0) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 0, 1) =
  Alien::Matrix(4, 5, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 0) =
  Alien::Matrix(5, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 1) =
  Alien::Matrix(5, 5, AlienTest::Environment::parallelMng());
  ASSERT_EQ(9, v.rowSpace().size());
  ASSERT_EQ(9, v.colSpace().size());
  auto& c00 = v(0, 0);
  ASSERT_EQ(4, c00.rowSpace().size());
  ASSERT_EQ(4, c00.colSpace().size());
  auto& c01 = v(0, 1);
  ASSERT_EQ(4, c01.rowSpace().size());
  ASSERT_EQ(5, c01.colSpace().size());
  auto& c10 = v(1, 0);
  ASSERT_EQ(5, c10.rowSpace().size());
  ASSERT_EQ(4, c10.colSpace().size());
  auto& c11 = v(1, 1);
  ASSERT_EQ(5, c11.rowSpace().size());
  ASSERT_EQ(5, c11.colSpace().size());
}

TEST(TestCompositeMatrix, CompositeResize)
{
  Alien::CompositeMatrix v;
  ASSERT_EQ(0, v.size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(0, v.rowSpace().size());
  ASSERT_EQ(0, v.colSpace().size());
  v.resize(2);
  Alien::CompositeElement(v, 0, 0) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 0, 1) =
  Alien::Matrix(4, 5, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 0) =
  Alien::Matrix(5, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 1) =
  Alien::Matrix(5, 5, AlienTest::Environment::parallelMng());
  ASSERT_EQ(9, v.rowSpace().size());
  ASSERT_EQ(9, v.colSpace().size());
  auto& c00 = v(0, 0);
  ASSERT_EQ(4, c00.rowSpace().size());
  ASSERT_EQ(4, c00.colSpace().size());
  auto& c01 = v(0, 1);
  ASSERT_EQ(4, c01.rowSpace().size());
  ASSERT_EQ(5, c01.colSpace().size());
  auto& c10 = v(1, 0);
  ASSERT_EQ(5, c10.rowSpace().size());
  ASSERT_EQ(4, c10.colSpace().size());
  auto& c11 = v(1, 1);
  ASSERT_EQ(5, c11.rowSpace().size());
  ASSERT_EQ(5, c11.colSpace().size());
}

TEST(TestCompositeMatrix, CompositeMultipleResize)
{
  Alien::CompositeMatrix v;
  ASSERT_EQ(0, v.size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(0, v.rowSpace().size());
  ASSERT_EQ(0, v.colSpace().size());
  v.resize(2);
  Alien::CompositeElement(v, 0, 0) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 0, 1) =
  Alien::Matrix(4, 5, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 0) =
  Alien::Matrix(5, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 1) =
  Alien::Matrix(5, 5, AlienTest::Environment::parallelMng());
  ASSERT_EQ(9, v.rowSpace().size());
  ASSERT_EQ(9, v.colSpace().size());
  {
    auto& c00 = v(0, 0);
    ASSERT_EQ(4, c00.rowSpace().size());
    ASSERT_EQ(4, c00.colSpace().size());
    auto& c01 = v(0, 1);
    ASSERT_EQ(4, c01.rowSpace().size());
    ASSERT_EQ(5, c01.colSpace().size());
    auto& c10 = v(1, 0);
    ASSERT_EQ(5, c10.rowSpace().size());
    ASSERT_EQ(4, c10.colSpace().size());
    auto& c11 = v(1, 1);
    ASSERT_EQ(5, c11.rowSpace().size());
    ASSERT_EQ(5, c11.colSpace().size());
  }
  v.resize(3);
  ASSERT_EQ(3, v.size());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      auto& c = v(i, j);
      // ASSERT_THROW(c.impl(), Alien::FatalErrorException);
      ASSERT_EQ(0, c.rowSpace().size());
      ASSERT_EQ(0, c.colSpace().size());
    }
  }
  Alien::CompositeElement(v, 0, 0) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 0, 1) =
  Alien::Matrix(4, 5, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 0) =
  Alien::Matrix(5, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 1) =
  Alien::Matrix(5, 5, AlienTest::Environment::parallelMng());
  ASSERT_EQ(9, v.rowSpace().size());
  ASSERT_EQ(9, v.colSpace().size());
  {
    auto& c00 = v(0, 0);
    ASSERT_EQ(4, c00.rowSpace().size());
    ASSERT_EQ(4, c00.colSpace().size());
    auto& c01 = v(0, 1);
    ASSERT_EQ(4, c01.rowSpace().size());
    ASSERT_EQ(5, c01.colSpace().size());
    auto& c02 = v(0, 2);
    ASSERT_EQ(0, c02.rowSpace().size());
    ASSERT_EQ(0, c02.colSpace().size());
    auto& c10 = v(1, 0);
    ASSERT_EQ(5, c10.rowSpace().size());
    ASSERT_EQ(4, c10.colSpace().size());
    auto& c11 = v(1, 1);
    ASSERT_EQ(5, c11.rowSpace().size());
    ASSERT_EQ(5, c11.colSpace().size());
    auto& c12 = v(1, 2);
    ASSERT_EQ(0, c12.rowSpace().size());
    ASSERT_EQ(0, c12.colSpace().size());
    auto& c20 = v(2, 0);
    ASSERT_EQ(0, c20.rowSpace().size());
    ASSERT_EQ(0, c20.colSpace().size());
    auto& c21 = v(2, 1);
    ASSERT_EQ(0, c21.rowSpace().size());
    ASSERT_EQ(0, c21.colSpace().size());
    auto& c22 = v(2, 2);
    ASSERT_EQ(0, c22.rowSpace().size());
    ASSERT_EQ(0, c22.colSpace().size());
  }
}

TEST(TestCompositeMatrix, TimeStampTest)
{
  Alien::CompositeMatrix v;
  auto* impl = v.impl();
  ASSERT_EQ(0, impl->timestamp());
  std::cout << "main ts = " << impl->timestamp() << std::endl;
  v.resize(2);
  Alien::CompositeElement(v, 0, 0) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 0, 1) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 0) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  Alien::CompositeElement(v, 1, 1) =
  Alien::Matrix(4, 4, AlienTest::Environment::parallelMng());
  ASSERT_EQ(0, impl->timestamp());
  std::cout << "main ts = " << impl->timestamp() << std::endl;
  auto& c0 = v(0, 0);
  auto* impl0 = c0.impl();
  ASSERT_EQ(0, impl0->timestamp());
  std::cout << "sub0 ts = " << impl0->timestamp() << std::endl;
  impl0->get<Alien::BackEnd::tag::simplecsr>(true);
  impl0->get<Alien::BackEnd::tag::simplecsr>(true);
  ASSERT_EQ(2, impl0->timestamp());
  std::cout << "sub0 ts = " << impl0->timestamp() << std::endl;
  ASSERT_EQ(2, impl->timestamp());
  std::cout << "main ts = " << impl->timestamp() << std::endl;
}
