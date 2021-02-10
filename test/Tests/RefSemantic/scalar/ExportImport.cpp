
#include <alien/AlienExternalPackages.h>
#include <alien/ref/AlienRefSemantic.h>
#include <alien/ref/AlienImportExport.h>

#include <Tests/Environment.h>

int
main(int argc, char** argv)
{
  // Pour initialiser MPI, les traces et
  // le gestionnaire de parallélisme
  Environment::initialize(argc, argv);

  auto* pm = Environment::parallelMng();
  auto* tm = Environment::traceMng();

  tm->info() << "Example Alien :";
  tm->info() << "Use of Alien::IVector readers / writers (RefSemanticMVHandlers API)";
  tm->info() << " ";
  tm->info() << "Start example";
  tm->info() << " ";

  // Le gestionnaire de trace est donné à Alien
  Alien::setTraceMng(tm);

  // On définit le niveau de verbosité
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);

  // Objects algébriques
  tm->info() << "define matrix";
  Alien::Space row_space(4, "RowSpace");
  Alien::Space col_space(4, "ColSpace");
  Alien::MatrixDistribution mdist(row_space, col_space, Environment::parallelMng());
  Alien::VectorDistribution vdist(col_space, Environment::parallelMng());
  Alien::Matrix A(mdist);

  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(7);
    builder.allocate();
    builder(0, 0) = 1.;
    builder(0, 1) = 6.;
    builder(0, 3) = 9.;
    builder(1, 1) = 2.;
    builder(2, 2) = 3.;
    builder(2, 3) = 4.;
    builder(3, 3) = 5.;
  }

  Alien::SystemWriter writer("matrix");
  writer.dump(A);

  Alien::SystemWriter writerHDF("matrix-hdf", "hdf5");
  writerHDF.dump(A);

  Alien::SystemWriter writerSMART("matrix-smart", "smart");
  writerSMART.dump(A);

  tm->info() << "* xe = 1";

  Alien::Vector xe = Alien::ones(4, pm);

  tm->info() << "=> Vector Distribution : " << xe.distribution();

  tm->info() << "* b = A * xe";

  Alien::Vector b(4, pm);

  Alien::PETScLinearAlgebra algebra;

  algebra.mult(A, xe, b);

  Alien::SystemWriter writer_b("matrix_b");
  writer_b.dump(A, b);

  Alien::SystemWriter writer_bHDF("matrix_b-hdf", "hdf5");
  writer_bHDF.dump(A, b);

  Alien::SystemWriter writer_bSMART("matrix_b-smart", "smart");
  writer_bSMART.dump(A, b);

  Alien::SolutionInfo sol_info(Alien::SolutionInfo::N2_RELATIVE2RHS_RES, 1e-16,
      "Unity solution, rhs obtained with petsc SpMV");

  Alien::SystemWriter writer_b_x("matrix_b_x");
  writer_b_x.dump(A, b, xe, sol_info);

  Alien::SystemWriter writer_b_xHDF("matrix_b_x-hdf", "hdf5");
  writer_b_xHDF.dump(A, b, xe, sol_info);

  Alien::SystemWriter writer_b_xSMART("matrix_b_x-smart", "smart");
  writer_b_xSMART.dump(A, b, xe, sol_info);

  /*
  Alien::SystemReader reader("matrix","ascii",pm);
  Alien::SystemReader readerHDF("matrix","hdf5",pm);

  Alien::Matrix B ;
  Alien::Matrix B_HDF;

  reader.read(B) ;

  reader.read(B_HDF);
  */
  return 0;
}
