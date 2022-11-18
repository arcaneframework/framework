#include <cassert>

#include <Tests/Options.h>

#include <alien/Alien.h>
#include <alien/AlienExternalPackages.h>
#include <alien/ref/AlienImportExport.h>
#include <alien/ref/AlienRefSemantic.h>
#include <alien/ref/data/scalar/RedistributedMatrix.h>
#include <alien/ref/data/scalar/RedistributedVector.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/kernels/redistributor/Redistributor.h>
#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_HYPRE
#include <alien/kernels/hypre/algebra/HypreLinearAlgebra.h>
#endif

// Méthode de construction de la matrice
extern void buildMatrix(
    Alien::Matrix& A, std::string const& filename, std::string const& format);

namespace {

enum algebraName
{
  simplecsr = 0,
  petsc = 1,
  hypre = 2,
  ifp = 3
};

void
testAlgebra(algebraName algebra, const Alien::ILinearAlgebra& alg,
    const Alien::IMatrix& A, const Alien::IVector& b, Alien::IVector& x)
{
  const Alien::Matrix& AasMatrix = static_cast<const Alien::Matrix&>(A);
  const Alien::Vector& basVector = static_cast<const Alien::Vector&>(b);
  Alien::Vector& xasVector = static_cast<Alien::Vector&>(x);
  const int lsize = basVector.distribution().localSize();
  const int gsize = basVector.distribution().globalSize();
  const int offset = basVector.distribution().offset();
  auto* pm = basVector.distribution().parallelMng();
  if (pm == nullptr)
    return;
  {
    alg.copy(basVector, xasVector);
    Alien::LocalVectorReader readerX(xasVector);
    Alien::LocalVectorReader readerB(basVector);
    for (int i = 0; i < lsize; ++i)
      assert(readerX[i] == readerB[i]);
  }
  const double dot = alg.dot(basVector, xasVector);
  assert(dot == gsize);
  if (algebra == algebraName::petsc) {
    const double norm0 = alg.norm0(xasVector);
    assert(norm0 == 1.);
    const double norm1 = alg.norm1(xasVector);
    assert(norm1 == gsize);
  }
  const double norm2 = alg.norm2(x);
  assert(norm2 == sqrt(dot));
  if (algebra != algebraName::hypre) {
    const double scal = 2.;
    alg.axpy(scal, b, x);
    {
      Alien::LocalVectorReader readerX(xasVector);
      Alien::LocalVectorReader readerB(basVector);
      for (int i = 0; i < lsize; ++i)
        assert(readerX[i] == 3. * readerB[i]);
    }
  }
  {
    alg.mult(AasMatrix, basVector, xasVector);
    Alien::LocalVectorReader readerX(xasVector);
    Alien::LocalVectorReader readerB(basVector);
    if (offset == 0)
      assert(readerX[0] == 1);
    for (int i = 1; i < lsize - 1; ++i)
      assert(readerX[i] == 0);
    if (offset + lsize == gsize)
      assert(readerX[lsize - 1] == 1);
  }
}
};

int
main(int argc, char** argv)
{
  return Environment::execute(argc, argv, [&] {

    auto* pm = Environment::parallelMng();
    auto* tm = Environment::traceMng();

    // Options pour ce test
    boost::program_options::options_description options;
    options.add_options()("help", "print usage")(
        "dump-on-screen", "dump algebraic objects on screen")(
        "dump-on-file", "dump algebraic objects on file")("size",
        boost::program_options::value<int>()->default_value(100),
        "size of problem")("file-name",
        boost::program_options::value<std::string>()->default_value("System"),
        "Input filename")("format",
        boost::program_options::value<std::string>()->default_value("ascii"),
        " format ascii or hdf5");

    // On récupère les options (+ celles de la configuration des solveurs)
    auto arguments = Environment::options(argc, argv, options);

    if (arguments.count("help")) {
      tm->info() << "Usage :\n" << options;
      return 1;
    }

    std::string redist_strategy = arguments["redist-strategy"].as<std::string>();

    std::string redist_method = arguments["redist-method"].as<std::string>();

    int size = arguments["size"].as<int>();

    std::string filename = arguments["file-name"].as<std::string>();
    std::string format = arguments["format"].as<std::string>();

    tm->info() << "Example Alien :";
    tm->info()
        << "Use of scalar builder (RefSemanticMVHandlers API) for Laplacian problem";
    tm->info() << " => solving linear system Ax = b";
    tm->info() << " * problem size = " << size;
    tm->info() << " ";
    tm->info() << "Start example...";
    tm->info() << " ";

    Alien::setTraceMng(tm);

    Alien::setVerbosityLevel(Alien::Verbosity::Debug);

    Alien::Matrix A(size, size, pm);

    tm->info() << "=> Matrix Distribution : " << A.distribution();

    buildMatrix(A, filename, format);
    size = A.rowSpace().size();

    Alien::Vector b = Alien::ones(size, pm);

    tm->info() << "=> Vector Distribution : " << b.distribution();

    Alien::Vector x(size, pm);

    tm->info() << "* x = A^-1 b";

    bool keep_proc = false;
    if (redist_strategy.compare("unique") == 0) {
      if (Environment::parallelMng()->commRank() == 0)
        keep_proc = true;
    } else {
      if ((Environment::parallelMng()->commRank() % 2) == 0)
        keep_proc = true;
    }

    Alien::Redistributor::Method method = Alien::Redistributor::dok;
    if(redist_method.compare("csr") == 0) {
      method = Alien::Redistributor::csr;
    }

    auto small_comm = Arccore::MessagePassing::mpSplit(Environment::parallelMng(), keep_proc);

    Alien::Redistributor redist(
        A.distribution().globalRowSize(), Environment::parallelMng(), small_comm, method);

    Alien::RedistributedMatrix Aa(A, redist);
    Alien::RedistributedVector bb(b, redist);
    Alien::RedistributedVector xx(x, redist);

    tm->info() << "Testing SimpleCSRLinearAlgebra";
    Alien::SimpleCSRLinearAlgebra csrAlg;
    testAlgebra(algebraName::simplecsr, csrAlg, Aa, bb, xx);
    tm->info() << "SimpleCSRLinearAlgebra OK";
    tm->info() << "Testing SimpleCSRLinearAlgebra2";
    Alien::SimpleCSRLinearAlgebra csrAlg2;
    testAlgebra(algebraName::simplecsr, csrAlg2, A, b, x);
    tm->info() << "SimpleCSRLinearAlgebra2 OK";
#ifdef ALIEN_USE_PETSC
    tm->info() << "Testing PETScLinearAlgebra";
    Alien::PETScLinearAlgebra petscAlg(Aa.distribution().parallelMng());
    testAlgebra(algebraName::petsc, petscAlg, Aa, bb, xx);
    tm->info() << "PETScLinearAlgebra OK";
    tm->info() << "Testing PETScLinearAlgebra2";
    Alien::PETScLinearAlgebra petscAlg2(A.distribution().parallelMng());
    testAlgebra(algebraName::petsc, petscAlg2, A, b, x);
    tm->info() << "PETScLinearAlgebra2 OK";
#endif
#ifdef ALIEN_USE_HYPRE
    tm->info() << "Testing HypreLinearAlgebra";
    Alien::HypreLinearAlgebra hypreAlg;
    testAlgebra(algebraName::hypre, hypreAlg, Aa, bb, xx);
    tm->info() << "HypreLinearAlgebra OK";
    tm->info() << "Testing HypreLinearAlgebra2";
    Alien::HypreLinearAlgebra hypreAlg2;
    testAlgebra(algebraName::hypre, hypreAlg2, A, b, x);
    tm->info() << "HypreLinearAlgebra2 OK";
#endif
#ifdef ALIEN_USE_IFPSOLVER
    tm->info() << "No algebra defined for IFP Solver";
#endif

    tm->info() << " ";
    tm->info() << "... example finished !!!";
    return 0;
  });
}
