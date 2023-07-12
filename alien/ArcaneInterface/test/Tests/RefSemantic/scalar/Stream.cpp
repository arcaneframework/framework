#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
extern Arccore::ITraceMng* traceMng();
} // namespace Environment

void
buildMatrix(Alien::Matrix& A, std::string const& filename, std::string const& format)
{
  auto* tm = Environment::traceMng();

  // Distributions calculée
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();
  int gsize = dist.globalRowSize();

  Alien::StreamMatrixBuilder stream(A);
  stream.setOrderRowColsOpt(true); // pour MCGSolver

  {
    tm->info() << "define matrix profile [1d-laplacian] with stream matrix profiler";

    Alien::StreamMatrixBuilder::Profiler& profiler = stream.getNewInserter();

    for (int irow = offset; irow < offset + lsize; ++irow) {
      profiler.addMatrixEntry(irow, irow);
      if (irow - 1 >= 0)
        profiler.addMatrixEntry(irow, irow - 1);
      if (irow + 1 < gsize)
        profiler.addMatrixEntry(irow, irow + 1);
    }
  }

  stream.allocate();

  tm->info() << "build matrix [1d-laplacian] with stream matrix filler";
  {
    Alien::StreamMatrixBuilder::Filler& builder = stream.getInserter(0);

    for (int irow = offset; irow < offset + lsize; ++irow) {
      builder.addData(2.);
      ++builder;
      if (irow - 1 >= 0) {
        builder.addData(-1.);
        ++builder;
      }
      if (irow + 1 < gsize) {
        builder.addData(-1.);
        ++builder;
      }
    }
  }
}
