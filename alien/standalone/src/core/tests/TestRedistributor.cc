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

// CEA or RefSemantic interface !
#if 0
#include <alien/data/MatrixData.h>
#include <alien/data/Scalar/Vector.h>

#include <alien/functional/Dump.h>
#include <alien/functional/Ones.h>

#include <alien/Builder/Scalar/MatrixProfiler.h>
#include <alien/Builder/Scalar/ProfiledMatrixBuilder.h>

#include <alien/handlers/VectorAccessors/Scalar/VectorWriter.h>

#include <alien/data/Scalar/RedistributedMatrix.h>
#include <alien/data/Scalar/RedistributedVector.h>

#include <alien/AlienExternalPackages.h>
#include <alien/AlienImportExport.h>

#endif // 0

#include <Environment.h>

using namespace Arccore;

TEST(TestRedistributor, RedistributorMatrix)
{
  Int32 rows = 3;
  Int32 cols = 3;
  Alien::MatrixDistribution mdist(rows, cols, AlienTest::Environment::parallelMng());
  Alien::Space row_space(mdist.globalRowSize(), "Space");
  Alien::Space col_space(mdist.globalColSize(), "Space");

  std::unique_ptr<Alien::MultiMatrixImpl> multimat(
  new Alien::MultiMatrixImpl(row_space.clone(), col_space.clone(), mdist.clone()));

  Alien::RedistributorMatrix mat(multimat.get());
  // mat.updateTargetPM(Environment::parallelMng());
}

TEST(TestRedistributor, NoDokRedistributorMatrix)
{
  Int32 rows = 3;
  Int32 cols = 3;
  Alien::MatrixDistribution mdist(rows, cols, AlienTest::Environment::parallelMng());
  Alien::Space row_space(mdist.globalRowSize(), "Space");
  Alien::Space col_space(mdist.globalColSize(), "Space");

  std::unique_ptr<Alien::MultiMatrixImpl> multimat(
  new Alien::MultiMatrixImpl(row_space.clone(), col_space.clone(), mdist.clone()));

  Alien::RedistributorMatrix mat(multimat.get(), false);
  // mat.updateTargetPM(Environment::parallelMng());
}

// This test requires RefSemantic API.
#if 0
TEST(TestRedistributor, Redistributor)
{
  int size = 12;
  Alien::Space row_space(size, "Space");
  Alien::Space col_space(size, "Space");

  Alien::MatrixDistribution mdist(row_space, col_space, Environment::parallelMng());
  int offset = mdist.rowOffset();
  int lsize = mdist.localRowSize();
  int gsize = mdist.globalRowSize();

  Alien::MatrixData A(row_space, col_space, mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);

  /*
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  Alien::DirectMatrixBuilder builder(std::move(A), tag);
  builder.reserve(3);
  builder.allocate();
  */

  Alien::MatrixProfiler profile(std::move(A));
  for(int irow = offset; irow < offset + lsize; ++irow)
  {
    profile.addMatrixEntry(irow, irow);
    if(irow - 1 >= 0)
      profile.addMatrixEntry(irow, irow - 1);
    if (irow + 1 < gsize)
      profile.addMatrixEntry(irow, irow + 1);
  }
  profile.allocate();
  A = profile.release();

  Alien::ProfiledMatrixBuilder builder(std::move(A), Alien::ProfiledMatrixOptions::eResetValues);
  for(int irow = offset; irow < offset + lsize; ++irow)
  {
    builder(irow,irow) = 2.;
    if(irow - 1 >= 0)
      builder(irow, irow - 1) = -1.;
      if (irow + 1 < gsize)
        builder(irow, irow + 1) = -1.;
  }

  builder.finalize();
  A = builder.release();
  
  Alien::Redistributor redist(mdist.globalRowSize(), Environment::parallelMng(), (Environment::parallelMng()->commRank() % 2 )==0);

  Alien::RedistributedMatrix Aa(A, redist);

  Alien::Vector b(gsize, Environment::parallelMng());
  Alien::Vector x(gsize, Environment::parallelMng());
  // Builder du vecteur
  Alien::VectorWriter writerB(b);
  Alien::VectorWriter writerX(x);
  
  // On remplit le vecteur
  for(int i = 0; i < lsize; ++i) {
    writerB[i+offset] = 1;//i+offset;
    writerX[i+offset] = 1;//i+offset;
  }

  Alien::RedistributedVector bb(b, redist);
  Alien::RedistributedVector xx(x, redist);

  Alien::SimpleCSRLinearAlgebra algebra;
  algebra.axpy(2.,bb, xx);
  algebra.mult(Aa, xx, bb);
}
#endif // 0
