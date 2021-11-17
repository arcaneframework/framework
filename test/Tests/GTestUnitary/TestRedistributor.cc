#include <gtest/gtest.h>

#include <alien/kernels/redistributor/Redistributor.h>
#include <alien/kernels/redistributor/RedistributorMatrix.h>

#include <alien/core/impl/MultiMatrixImpl.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

using namespace Arccore;

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

TEST(TestRedistributor, RedistributorMatrix)
{
  Arccore::Int32 rows = 3;
  Arccore::Int32 cols = 3;
  Alien::MatrixDistribution mdist(rows, cols, Environment::parallelMng());
  Alien::Space row_space(mdist.globalRowSize(), "Space");
  Alien::Space col_space(mdist.globalColSize(), "Space");

  std::unique_ptr<Alien::MultiMatrixImpl> multimat(
      new Alien::MultiMatrixImpl(row_space.clone(), col_space.clone(), mdist.clone()));

  Alien::RedistributorMatrix mat(multimat.get());
  // mat.updateTargetPM(Environment::parallelMng());
}

// This test requires IFPEN API.
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
