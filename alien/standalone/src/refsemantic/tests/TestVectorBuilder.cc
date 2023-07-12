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

TEST(TestVectorBuilder, ReleaseTest)
{
  Alien::Vector v(3, AlienTest::Environment::parallelMng());
  {
    Alien::VectorWriter writer(v);
  }
  ASSERT_EQ(3, v.space().size());
}

TEST(TestVectorBuilder, WriterTest)
{
  Alien::Vector v(3, AlienTest::Environment::parallelMng());
  {
    Alien::VectorWriter writer(v);
    writer[0] = 0.;
    writer[1] = 1.;
    writer[2] = 2.;
  }
  ASSERT_EQ(3, v.space().size());
}

TEST(TestVectorBuilder, ReaderWriterTest)
{
  Alien::Vector v(3, AlienTest::Environment::parallelMng());
  {
    Alien::VectorWriter writer(v);
    writer[0] = 0.;
    writer[1] = 1.;
    writer[2] = 2.;
  }
  ASSERT_EQ(3, v.space().size());
  {
    Alien::LocalVectorReader reader(v);
    ASSERT_EQ(0., reader[0]);
    ASSERT_EQ(1., reader[1]);
    ASSERT_EQ(2., reader[2]);
    ASSERT_EQ(3, reader.size());
  }
}
