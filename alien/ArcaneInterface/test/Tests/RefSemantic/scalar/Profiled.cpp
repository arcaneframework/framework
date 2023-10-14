#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Alien::ITraceMng* traceMng();
}

void
buildMatrix(Alien::Matrix& A, [[maybe_unused]] std::string const& filename,
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
    Alien::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);

    for (int irow = offset; irow < offset + lsize; ++irow) {
      builder(irow, irow) = 2.;
      if (irow - 1 >= 0)
        builder(irow, irow - 1) = -1.;
      if (irow + 1 < gsize)
        builder(irow, irow + 1) = -1.;
    }
  }
}
