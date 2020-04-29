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

#include "DefaultToolsForTesting.h"

#include <alien/UserObjects/Accessor/VectorReader.h>
#include <alien/UserObjects/Accessor/VectorWriter.h>
#include <alien/UserObjects/Builder/Scalar/DirectMatrixBuilder.h>
#include <alien/UserObjects/data/MatrixData.h>
#include <alien/UserObjects/data/Space.h>
#include <alien/UserObjects/data/VectorData.h>
#include <alien/core/kernels/simple_csr/Algebra/SimpleCSRLinearAlgebra.h>
#include <alien/expression/MatrixExp.h>
#include <alien/expression/VectorExp.h>
#include <alien/utils/Precomp.h>
#include <iostream>

TEST(TestMatrixIO, test)
{
  Testing::CoreTools t;
  auto& mdist = t.matrixDistribution(3, 4);
  auto& vdist = t.vectorDistribution(4);
  Alien::Space row_space(3, "RowSpace");
  Alien::Space col_space(4, "ColSpace");
  Alien::MatrixData A(row_space, col_space, mdist);
  ASSERT_FALSE(A.isNull());
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);
  auto tag = Alien::DirectMatrixBuilder::eResetValues;
  Alien::DirectMatrixBuilder builder(std::move(A), tag);
  builder.reserve(5);
  builder.allocate();
  builder(0, 0) = 1.;
  builder(1, 1) = 1.;
  builder(2, 2) = 1.;
  builder(2, 3) = 1.;
  builder.finalize();
  A = builder.release();

#if 0
  // check with spmv
  Alien::SimpleCSRLinearAlgebra Alg;
  Alien::VectorData X(col_space, vdist);
  Alien::LocalVectorWriter writer(std::move(X));
  writer[0] = 1.;
  writer[1] = 1.;
  writer[2] = 1.;
  writer[3] = 1.;
  X = writer.release();
  Alien::VectorData R(row_space, t.vectorDistribution(3));
  Alien::VectorExp vX(X);
  Alien::MatrixExp vA(A);
  Alg.mult(vA, vX, R);
  Alien::VectorExp vR(R);
  Alien::LocalVectorReader reader(vR);
  std::cout << reader[0] << std::endl;
  std::cout << reader[1] << std::endl;
  std::cout << reader[2] << std::endl;
#endif
}
