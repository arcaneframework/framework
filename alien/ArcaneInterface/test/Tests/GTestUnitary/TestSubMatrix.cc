#include "gtest/gtest.h"

#include <alien/Alien.h>

namespace Environment {
extern Alien::IParallelMng* parallelMng();
}

TEST(TestSubMatrix, RangeExtraction)
{

  // TODO

  //  std::cout << "Test submatrix\n";
  //
  //  auto* pm = Environment::parallelMng();
  //
  //  auto comm_size = pm->commSize();
  //
  //  auto nrows = comm_size > 1 ? 20*comm_size : 40;
  //
  //  Alien::MatrixDistribution dist(nrows, nrows, pm);
  //
  //  Alien::Space space(nrows,"TestSpace");
  //  Alien::MatrixData matrixA(space,dist);
  //  Alien::MatrixData matrixB(space,dist);
  //
  //  int bd = 2;
  //  double fii = 1;
  //  double fij = 1;
  //  double diag = fii +2*bd*fij;
  //  double offdiag = fij;
  //  int offset = dist.rowOffset();
  //  int local_size = dist.localRowSize();
  //  {
  //    Alien::DirectMatrixBuilder builder(std::move(matrixA),
  //    Alien::DirectMatrixOptions::eResetValues);
  //    builder.reserve(5);
  //    builder.allocate();
  //    for(int irow=offset;irow<offset+local_size;++irow)
  //    {
  //      builder.setData(irow,irow,diag);
  //      for(int j=1;j<bd;++j)
  //      {
  //        if(irow-j>=0)
  //          builder.setData(irow,irow-j,-offdiag/j);
  //        if(irow+j<nrows)
  //          builder.setData(irow,irow+j,-offdiag/j);
  //      }
  //    }
  //
  //    builder.finalize();
  //    matrixA = builder.release();
  //  }
  //  // Extract diagonal blocks of the matrix
  //  Alien::ExtractionIndices indices(0,20,0,20);
  //  Alien::MatrixData subMatrixA = Alien::SubMatrix::Extract(matrixA, indices);
  //  Alien::ExtractionIndices indices2(20,20,20,20);
  //  Alien::MatrixData subMatrixB = Alien::SubMatrix::Extract(matrixA, indices2);
}
