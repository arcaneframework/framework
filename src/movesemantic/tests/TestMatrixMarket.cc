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
#include <alien/distribution/VectorDistribution.h>

#include <alien/move/data/MatrixData.h>

#include <Environment.h>

TEST(TestMatrixMarket, Functional)
{
  auto mat = Alien::Move::readFromMatrixMarket(AlienTest::Environment::parallelMng(), "simple.mtx");
  ASSERT_EQ(mat.rowSpace().size(), 25);
  ASSERT_EQ(mat.colSpace().size(), 25);
}
