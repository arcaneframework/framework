#include <gtest/gtest.h>

#include <alien/Alien.h>
#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestMatrixDirectBuilder, ConstructorWithSpaces)
{
  Alien::Space row_space(3, "RowSpace");
  Alien::Space col_space(4, "ColSpace");
  Alien::MatrixDistribution mdist(row_space, col_space, Environment::parallelMng());
  Alien::VectorDistribution vdist(col_space, Environment::parallelMng());
  Alien::Matrix A(mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(5);
    builder.allocate();
    builder(0, 0) = 1.;
    builder(1, 1) = 1.;
    builder(2, 2) = 1.;
    builder(2, 3) = 1.;
  }
  // check with spmv
  Alien::SimpleCSRLinearAlgebra Alg;
  Alien::Vector X(vdist);
  {
    Alien::LocalVectorWriter writer(X);
    writer[0] = 1.;
    writer[1] = 1.;
    writer[2] = 1.;
    writer[3] = 1.;
  }
  Alien::VectorDistribution vdist2(row_space, Environment::parallelMng());
  Alien::Vector R(vdist2);
  Alg.mult(A, X, R);
  {
    Alien::LocalVectorReader reader(R);
    std::cout << reader[0] << std::endl;
    std::cout << reader[1] << std::endl;
    std::cout << reader[2] << std::endl;
  }
}
