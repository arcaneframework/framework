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

#include <Environment.h>

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/Space.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/kernels/composite/CompositeMultiVectorImpl.h>

TEST(TestBackEnds, Constructor)
{
  Alien::VectorDistribution dist(3, AlienTest::Environment::parallelMng());
  Alien::Space sp(3);
  Alien::MultiVectorImpl impl(std::make_shared<Alien::Space>(sp), dist.clone());
  ASSERT_EQ(nullptr, impl.block());
  ASSERT_EQ(sp, impl.space());
  ASSERT_EQ(dist, impl.distribution());
}

TEST(TestBackEnds, CompositeVector)
{
  Alien::CompositeKernel::MultiVectorImpl impl;
  ASSERT_EQ(nullptr, impl.block());
  ASSERT_EQ(Alien::Space(), impl.space());
  ASSERT_EQ(Alien::VectorDistribution(), impl.distribution());
}
