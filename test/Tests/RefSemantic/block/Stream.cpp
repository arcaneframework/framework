

#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::ITraceMng* traceMng();
}

void
buildMatrix(Alien::BlockMatrix& A, std::string const& filename, std::string const& format)
{
  auto* tm = Environment::traceMng();

  // Distributions calculée
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();
  int gsize = dist.globalRowSize();

  Alien::StreamMatrixBuilder stream(A);

  tm->info() << "define matrix profile [1d-laplacian] with stream matrix builder";
  {
    // Création d'un inserter pour le profile
    // Nb: auto ne fonctionne pas
    Alien::StreamMatrixBuilder::Profiler& profiler = stream.getNewInserter();

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

  tm->info() << "build matrix [1d-laplacian] with stream matrix builder";
  {
    // Création d'un builder pour les coefficients
    // Nb: auto ne fonctionne pas
    Alien::StreamMatrixBuilder::Filler& builder = stream.getInserter(0);
    const int block_size = A.block().size();
    Alien::UniqueArray<double> values2d(block_size * block_size),
        values2dExtraDiag(block_size * block_size);
    values2d.fill(0.);
    values2dExtraDiag.fill(0.);
    for (int i = offset; i < offset + lsize; ++i) {
      for (int j = 0; j < block_size; ++j) {
        values2d[j * block_size + j] = 4;
        values2dExtraDiag[j * block_size + j] = -1;
        if (j - 1 >= 0)
          values2d[j * block_size + j - 1] = -1;
        if (j + 1 < block_size)
          values2d[j * block_size + j + 1] = -1;
      }
      builder.addBlockData(values2d);
      ++builder;
      if (i - 1 >= 0) {
        builder.addBlockData(values2dExtraDiag);
        ++builder;
      }
      if (i + 1 < gsize) {
        builder.addBlockData(values2dExtraDiag);
        ++builder;
      }
    }
  }
}
