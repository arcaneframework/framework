/*
 * Copyright 2021 IFPEN-CEA
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
 *  SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

#include <alien/move/data/MatrixData.h>
#include <alien/move/handlers/scalar/DoKDirectMatrixBuilder.h>
#include <alien/move/data/VectorData.h>
#include <alien/move/handlers/scalar/VectorWriter.h>
#include <alien/move/handlers/scalar/VectorReader.h>

#include <Environment.h>
#include <alien/core/backend/LinearAlgebra.h>

TEST(TestDoKDirectMatrixBuilder, Fill)
{
  Alien::Space row_space(5, "RowSpace");
  Alien::Space col_space(4, "ColSpace");
  Alien::MatrixDistribution mdist(
  row_space, col_space, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(col_space, AlienTest::Environment::parallelMng());
  Alien::Move::MatrixData A(mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);
  {
    auto builder = Alien::Move::DoKDirectMatrixBuilder(std::move(A));

    // With DokDirectMatrixBuilder, this code is valid for all ranks.
    // We can fill non-local non-zeros.
    ASSERT_TRUE(builder.contribute(0, 0, 1.));
    ASSERT_TRUE(builder.contribute(1, 1, 1.));
    auto out = builder.contribute(1, 1, 2.);
    ASSERT_TRUE(out);
    ASSERT_EQ(out.value(), 3.);
    ASSERT_TRUE(builder.contribute(2, 2, 1.));
    ASSERT_TRUE(builder.contribute(2, 3, 1.));
    A = builder.release();
    ASSERT_FALSE(builder.contribute(3, 3, 1.));
  }
}
