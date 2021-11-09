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
#include <alien/core/backend/LinearAlgebra.h>

TEST(TestMatrixDirectBuilder, ConstructorWithSpaces)
{
  Alien::Space row_space(3, "RowSpace");
  Alien::Space col_space(4, "ColSpace");
  Alien::MatrixDistribution mdist(
  row_space, col_space, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(col_space, AlienTest::Environment::parallelMng());
  Alien::Matrix A(mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(
    A, tag, Alien::DirectMatrixOptions::SymmetricFlag::eUnSymmetric);
    builder.reserve(5);
    builder.allocate();
    builder(0, 0) = 1.;
    builder(1, 1) = 1.;
    builder(2, 2) = 1.;
    builder(2, 3) = 1.;
  }
  // check with spmv
  Alien::LinearAlgebra<Alien::BackEnd::tag::simplecsr> Alg(vdist.parallelMng());
  Alien::Vector X(vdist);
  {
    Alien::LocalVectorWriter writer(X);
    writer[0] = 1.;
    writer[1] = 2.;
    writer[2] = 3.;
    writer[3] = 4.;
  }
  Alien::VectorDistribution vdist2(row_space, AlienTest::Environment::parallelMng());
  Alien::Vector R(vdist2);
  Alg.mult(A, X, R);
  {
    Alien::LocalVectorReader reader(R);
    std::cout << reader[0] << std::endl;
    std::cout << reader[1] << std::endl;
    std::cout << reader[2] << std::endl;

    ASSERT_EQ(reader[0], 1.);
    ASSERT_EQ(reader[1], 2.);
    ASSERT_EQ(reader[2], 7.);
  }
}
