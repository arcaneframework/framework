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

#include <alien/kernels/redistributor/Redistributor.h>
#include <alien/kernels/redistributor/RedistributorMatrix.h>

#include <alien/core/impl/MultiMatrixImpl.h>

#include <alien/ref/data/scalar/Vector.h>
#include <alien/ref/data/scalar/Matrix.h>
#include <alien/ref/data/scalar/RedistributedMatrix.h>
#include <alien/ref/data/scalar/RedistributedVector.h>

#include <alien/ref/handlers/scalar/DirectMatrixBuilder.h>

#include <alien/ref/handlers/scalar/VectorWriter.h>

#include <Environment.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

using namespace Arccore;

// This test requires RefSemantic API.
TEST(TestRedistributorRef, Redistributor)
{
  int size = 12;
  Alien::Space row_space(size, "Space");
  Alien::Space col_space(size, "Space");

  Alien::MatrixDistribution mdist(row_space, col_space, AlienTest::Environment::parallelMng());
  int offset = mdist.rowOffset();
  int lsize = mdist.localRowSize();
  int gsize = mdist.globalRowSize();

  Alien::Matrix A(mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);

  Alien::DirectMatrixBuilder builder(A, Alien::DirectMatrixOptions::ResetFlag::eResetAllocation);
  builder.allocate(); // FIXME Important, else inserting leads to segfault
  for (int irow = offset; irow < offset + lsize; ++irow) {
    builder(irow, irow) = 2.;
    if (irow - 1 >= 0)
      builder(irow, irow - 1) = -1.;
    if (irow + 1 < gsize)
      builder(irow, irow + 1) = -1.;
  }
  builder.finalize();

  auto small_comm = Arccore::MessagePassing::mpSplit(AlienTest::Environment::parallelMng(), (AlienTest::Environment::parallelMng()->commRank() % 2) == 0);

  Alien::Redistributor redist(mdist.globalRowSize(), AlienTest::Environment::parallelMng(),
                              small_comm);

  Alien::RedistributedMatrix Aa(A, redist);

  Alien::Vector b(gsize, AlienTest::Environment::parallelMng());
  Alien::Vector x(gsize, AlienTest::Environment::parallelMng());
  // Builder du vecteur
  Alien::VectorWriter writerB(b);
  Alien::VectorWriter writerX(x);

  // On remplit le vecteur
  for (int i = 0; i < lsize; ++i) {
    writerB[i + offset] = 1; //i+offset;
    writerX[i + offset] = 1; //i+offset;
  }

  Alien::RedistributedVector bb(b, redist);
  Alien::RedistributedVector xx(x, redist);

  Alien::SimpleCSRLinearAlgebra algebra;
  algebra.axpy(2., bb, xx);
  algebra.mult(Aa, xx, bb);
}
