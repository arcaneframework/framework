
#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::ITraceMng* traceMng();
}

void
buildMatrix(Alien::VBlockMatrix& A)
{
  auto* tm = Environment::traceMng();

  // Distributions calculée
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();
  int gsize = dist.globalRowSize();

  Alien::StreamVBlockMatrixBuilder stream(A);

  tm->info() << "define matrix profile [1d-laplacian] with stream vblock matrix builder";
  {
    // Création d'un inserter pour le profile
    // Nb: auto ne fonctionne pas
    Alien::StreamVBlockMatrixBuilder::Profiler& profiler = stream.getNewInserter();

    for (int irow = offset; irow < offset + lsize; ++irow) {
      profiler.addMatrixEntry(irow, irow);
      if (irow - 1 >= 0)
        profiler.addMatrixEntry(irow, irow - 1);
      if (irow + 1 < gsize)
        profiler.addMatrixEntry(irow, irow + 1);
    }
  }

  // Allocation du profile
  stream.allocate();

  tm->info() << "build matrix [1d-laplacian] with stream vblock matrix builder";
  {
    // Création d'un builder pour les coefficients
    // Nb: auto ne fonctionne pas
    Alien::StreamVBlockMatrixBuilder::Filler& builder = stream.getInserter(0);

    for (int i = offset; i < offset + lsize; ++i) {
      const int block_size = A.vblock().size(i);
      Alien::UniqueArray2<double> values2d, values2dExtraDiag;
      values2d.resize(block_size, block_size);
      values2dExtraDiag.resize(block_size, block_size);
      values2d.fill(0.);
      values2dExtraDiag.fill(0.);

      for (int j = 0; j < block_size; ++j) {
        values2d[j][j] = 4;
        values2dExtraDiag[j][j] = -1;
        if (j - 1 >= 0)
          values2d[j][j - 1] = -1;
        if (j + 1 < block_size)
          values2d[j][j + 1] = -1;
      }
      builder.addBlockData(values2d.view());
      ++builder;
      if (i - 1 >= 0) {
        builder.addBlockData(values2dExtraDiag.view());
        ++builder;
      }
      if (i + 1 < gsize) {
        builder.addBlockData(values2dExtraDiag.view());
        ++builder;
      }
    }
  }
}
