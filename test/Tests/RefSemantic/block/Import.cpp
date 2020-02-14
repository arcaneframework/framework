
#include <ALIEN/Alien-ImportExport.h>
#include <ALIEN/Alien-RefSemantic.h>

namespace Environment {
extern Arccore::ITraceMng* traceMng();
}

void buildMatrix(Alien::BlockMatrix& A,std::string const& filename, std::string const& format)
{
  auto* tm = Environment::traceMng();

  tm->info() << "build matrix with MatrixFileImporter";
  {
    Alien::SystemReader matrix_reader(filename,format,A.distribution().parallelMng());
    matrix_reader.read(A) ;
  }
}
