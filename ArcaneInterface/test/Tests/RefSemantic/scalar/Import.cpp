#include <string>
#include <alien/ref/AlienImportExport.h>
#include <alien/ref/AlienRefSemantic.h>

namespace Environment {
extern Arccore::ITraceMng* traceMng();
}
void
buildMatrix(Alien::Matrix& A, std::string const& filename, std::string const& format)
{
  auto* tm = Environment::traceMng();

  tm->info() << "build matrix with MatrixFileImporter";
  {
    Alien::SystemReader matrix_reader(filename, format, A.distribution().parallelMng());
    matrix_reader.read(A);
  }
}
