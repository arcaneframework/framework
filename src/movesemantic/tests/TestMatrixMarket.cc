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

#include <cmath>
#include <gtest/gtest.h>

#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

#include <alien/move/data/MatrixData.h>
#include <alien/move/data/VectorData.h>

#include <alien/move/handlers/scalar/VectorReader.h>

#include <Environment.h>

TEST(TestMatrixMarket, MatrixAlone)
{
  auto mat = Alien::Move::readFromMatrixMarket(AlienTest::Environment::parallelMng(), "simple.mtx");
  ASSERT_EQ(mat.rowSpace().size(), 25);
  ASSERT_EQ(mat.colSpace().size(), 25);
}

void check_vect_simple_values(const Alien::Move::VectorData& vect)
{
  // Check vector values
  Alien::Move::LocalVectorReader local_vect(vect);
  auto offset = vect.distribution().offset();
  for (int i = 0; i < local_vect.size(); i++) {
    ASSERT_EQ(i + offset + 1, ::floor(local_vect[i]));
  }
}

TEST(TestMatrixMarket, VectorAlone)
{
  Alien::Space s(25);
  Alien::VectorDistribution vd(s, AlienTest::Environment::parallelMng());
  auto vect = Alien::Move::readFromMatrixMarket(vd, "simple_rhs.mtx");
  check_vect_simple_values(vect);
}

TEST(TestMatrixMarket, MatrixVector)
{
  auto mat = Alien::Move::readFromMatrixMarket(AlienTest::Environment::parallelMng(), "simple.mtx");
  auto vect = Alien::Move::readFromMatrixMarket(mat.distribution().rowDistribution(), "simple_rhs.mtx");
  check_vect_simple_values(vect);
}
