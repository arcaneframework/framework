#include <gtest/gtest.h>

#include "DefaultToolsForTesting.h"

#include <alien/utils/Precomp.h>
#include <ALIEN/UserObjects/Data/MatrixData.h>
#include <ALIEN/UserObjects/Data/VectorData.h>
#include <ALIEN/Expression/VectorExp.h>
#include <ALIEN/Expression/MatrixExp.h>
#include <ALIEN/UserObjects/Data/Space.h>
#include <ALIEN/UserObjects/Builder/Scalar/DirectMatrixBuilder.h>
#include <ALIEN/UserObjects/Accessor/VectorReader.h>
#include <ALIEN/UserObjects/Accessor/VectorWriter.h>
#include <ALIEN/Core/Kernels/SimpleCSR/algebra/SimpleCSRLinearAlgebra.h>

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
