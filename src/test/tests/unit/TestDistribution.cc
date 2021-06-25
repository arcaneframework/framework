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

#include <alien/distribution/MatrixDistribution.h>
#include <alien/kernels/redistributor/Redistributor.h>

TEST(TestDistribution, VectorDefaultConstructor)
{
  Alien::VectorDistribution vd;
  ASSERT_FALSE(vd.isParallel());
}

TEST(TestDistribution, VectorGlobalSizeConstructor)
{
  auto* pm = AlienTest::Environment::parallelMng();
  Alien::VectorDistribution vd(10, pm);
  ASSERT_EQ(vd.isParallel(), pm->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), pm);
  ASSERT_EQ(10, vd.globalSize());
  auto gsize = Arccore::MessagePassing::mpAllReduce(
  pm, Arccore::MessagePassing::ReduceSum, vd.localSize());
  ASSERT_EQ(10, gsize);
}

TEST(TestDistribution, VectorGlobalLocalSizeConstructor)
{
  auto* pm = AlienTest::Environment::parallelMng();
  auto np = pm->commSize();
  auto rk = pm->commRank();
  auto global_size = 2 * np;
  Alien::VectorDistribution vd(global_size, 2, pm);
  ASSERT_EQ(vd.isParallel(), pm->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), pm);
  ASSERT_EQ(global_size, vd.globalSize());
  ASSERT_EQ(2, vd.localSize());
  ASSERT_EQ(2 * rk, vd.offset());
}

TEST(TestDistribution, MatrixDefaultConstructor)
{
  Alien::MatrixDistribution vd;
  ASSERT_FALSE(vd.isParallel());
}

TEST(TestDistribution, MatrixGlobalSizeConstructor)
{
  auto* pm = AlienTest::Environment::parallelMng();
  Alien::MatrixDistribution vd(10, 5, pm);
  ASSERT_EQ(vd.isParallel(), pm->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), pm);
  ASSERT_EQ(10, vd.globalRowSize());
  auto gsize = Arccore::MessagePassing::mpAllReduce(
  pm, Arccore::MessagePassing::ReduceSum, vd.localRowSize());
  ASSERT_EQ(10, gsize);
  ASSERT_EQ(5, vd.globalColSize());
  // ASSERT_EQ(5, vd.localColSize()); //FIXME: Why ?
  // ASSERT_EQ(0, vd.colOffset()); //FIXME: Why ?
}

TEST(TestDistribution, MatrixGlobalLocalSizeConstructor)
{
  auto* pm = AlienTest::Environment::parallelMng();
  auto np = pm->commSize();
  auto rk = pm->commRank();
  auto global_size = 2 * np;
  Alien::MatrixDistribution vd(global_size, 5, 2, pm);
  ASSERT_EQ(vd.isParallel(), pm->commSize() > 1);
  ASSERT_EQ(vd.parallelMng(), pm);
  ASSERT_EQ(global_size, vd.globalRowSize());
  ASSERT_EQ(2, vd.localRowSize());
  ASSERT_EQ(2 * rk, vd.rowOffset());
  ASSERT_EQ(5, vd.globalColSize());
  // ASSERT_EQ(5, vd.localColSize()); //FIXME: Why ?
}
