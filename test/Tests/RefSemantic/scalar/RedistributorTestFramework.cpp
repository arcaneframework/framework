#include <Tests/Options.h>

#include <alien/Alien.h>
#include <alien/AlienExternalPackages.h>
#include <alien/ref/AlienImportExport.h>
#include <alien/ref/AlienRefSemantic.h>
#include <alien/ref/data/scalar/RedistributedMatrix.h>
#include <alien/ref/data/scalar/RedistributedVector.h>
#include <alien/kernels/redistributor/Redistributor.h>

#include <Tests/Solver.h>

// Méthode de construction de la matrice
extern void buildMatrix(
    Alien::Matrix& A, std::string const& filename, std::string const& format);

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

    std::string redist_strategy = arguments["redist-strategy"].as<std::string>();

    if (arguments.count("help")) {
      tm->info() << "Usage :\n" << options;
      return 1;
    }

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

#ifdef ALIEN_USE_PETSC
    if (arguments.count("dump-on-screen")) {
      tm->info() << "dump A on screen";
      Alien::dump(A);
    }
#endif // ALIEN_USE_PETSC

    tm->info() << "* xe = 1";

    Alien::Vector xe = Alien::ones(size, pm);

    tm->info() << "=> Vector Distribution : " << xe.distribution();

#ifdef ALIEN_USE_PETSC
    if (arguments.count("dump-on-screen")) {
      tm->info() << "dump exact solution xe";
      Alien::dump(xe);
    }
#endif // ALIEN_USE_PETSC
#ifdef ALIEN_USE_PETSC
    tm->info() << "* b = A * xe";

    Alien::Vector b(size, pm);

    Alien::SimpleCSRLinearAlgebra algebra;

    algebra.mult(A, xe, b);

    if (arguments.count("dump-on-screen")) {
      tm->info() << "dump b";
      Alien::dump(b);
    }

    if (arguments.count("dump-on-file")) {
      tm->info() << "dump A, b";
      Alien::SystemWriter system_writer("System-out", format, pm);
      system_writer.dump(A, b);
    }

    tm->info() << " ** [solver package=" << arguments["solver-package"].as<std::string>()
               << "]";

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

    auto small_comm = Arccore::MessagePassing::mpSplit(Environment::parallelMng(), keep_proc);

    Alien::Redistributor redist(
        A.distribution().globalRowSize(), Environment::parallelMng(), small_comm);

    Alien::RedistributedMatrix Aa(A, redist);
    Alien::RedistributedVector bb(b, redist);
    Alien::RedistributedVector xx(x, redist);

    auto solver = Environment::createSolver(arguments);
    solver->updateParallelMng(Aa.distribution().parallelMng());
    solver->init();
    solver->solve(Aa, bb, xx);

    tm->info() << "* r = Ax - b";

    Alien::Vector r(size, pm);

    {
      Alien::Vector tmp(size, pm);
      tm->info() << "t = Ax";
      algebra.mult(A, x, tmp);
      tm->info() << "r = t";
      algebra.copy(tmp, r);
      tm->info() << "r -= b";
      algebra.axpy(-1., b, r);
    }

    auto norm = algebra.norm2(r);

    tm->info() << " => ||r|| = " << norm;

    if (arguments.count("dump-on-screen")) {
      tm->info() << "dump solution";
      Alien::dump(x);
    }

    tm->info() << "* r = || x - xe ||";

    {
      tm->info() << "r = x";
      algebra.copy(x, r);
      tm->info() << "r -= xe";
      algebra.axpy(-1., xe, r);
    }

    tm->info() << " => ||r|| = " << norm;

    if (norm > arguments["tol"].as<double>())
      tm->fatal() << "Error, residual norm is to high (" << norm << " vs. "
                  << arguments["tol"].as<double>() << ")\n";

    tm->info() << " ";
    tm->info() << "... example finished !!!";
#endif // ALIEN_USE_PETSC
    return 0;
  });
}
