

#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Alien::ITraceMng* traceMng();
}

void
buildMatrix(Alien::BlockMatrix& A, [[maybe_unused]] std::string const& filename,
            [[maybe_unused]] std::string const& format)
{
  auto* tm = Environment::traceMng();

  // Distributions calculée
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();
  int gsize = dist.globalRowSize();

  tm->info() << "define matrix profile [1d-laplacian] with matrix profiler";
  {
    Alien::MatrixProfiler profiler(A);

    for (int irow = offset; irow < offset + lsize; ++irow) {
      profiler.addMatrixEntry(irow, irow);
      if (irow - 1 >= 0)
        profiler.addMatrixEntry(irow, irow - 1);
      if (irow + 1 < gsize)
        profiler.addMatrixEntry(irow, irow + 1);
    }
  }

  tm->info() << "build matrix [1d-laplacian] with profiled matrix builder";
  {
    Alien::ProfiledBlockMatrixBuilder builder(
        A, Alien::ProfiledBlockMatrixBuilderOptions::eResetValues);
    Alien::UniqueArray2<double> values2d, values2dExtraDiag;
    const int block_size = A.block().size();
    values2d.resize(block_size, block_size);
    values2dExtraDiag.resize(block_size, block_size);
    values2d.fill(0.);
    values2dExtraDiag.fill(0.);
    for (int i = offset; i < offset + lsize; ++i) {
      for (int j = 0; j < block_size; ++j) {
        values2d[j][j] = 4;
        values2dExtraDiag[j][j] = -1;
        if (j - 1 >= 0)
          values2d[j][j - 1] = -1;
        if (j + 1 < block_size)
          values2d[j][j + 1] = -1;
      }
      builder(i, i) = values2d.view();
      if (i - 1 >= 0)
        builder(i, i - 1) = values2dExtraDiag.view();
      if (i + 1 < gsize)
        builder(i, i + 1) = values2dExtraDiag.view();
    }
  }
}
