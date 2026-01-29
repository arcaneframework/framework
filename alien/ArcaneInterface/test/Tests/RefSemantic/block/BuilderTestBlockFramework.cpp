#include <Tests/Options.h>

#include <alien/ref/AlienImportExport.h>
#include <alien/ref/AlienRefSemantic.h>

#include <Tests/Solver.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/hypre/algebra/HypreLinearAlgebra.h>

template<typename AlgebraT>
void test(Alien::ITraceMng* tm,const std::string& format,
    Arccore::MessagePassing::IMessagePassingMng *pm,
    boost::program_options::variables_map& arguments,
    AlgebraT& algebra,int size,const Alien::Block& block,
    const Alien::BlockMatrix& A,const Alien::BlockVector& xe,Alien::BlockVector& b)
{

  algebra.mult(A, xe, b);

#ifdef ALIEN_USE_PETSC
  if (arguments.count("dump-on-screen")) {
    tm->info() << "dump b";
    Alien::dump(b);
  }
#endif

#ifdef ALIEN_USE_PETSC
  if (arguments.count("dump-on-file")) {
    tm->info() << "dump A, b";
    Alien::SystemWriter system_writer("System-out", format, pm);
    system_writer.dump(A, b);
  }
#endif

  tm->info() << " ** [solver package=" << arguments["solver-package"].as<std::string>()
             << "]";

  Alien::BlockVector x(size, block, pm);

  tm->info() << "* x = A^-1 b";

  auto solver = Environment::createSolver(arguments);
  solver->init();
  solver->solve(A, b, x);

  tm->info() << "* r = Ax - b";

  Alien::BlockVector r(size, block, pm);

  {
    Alien::BlockVector tmp(size, block, pm);
    tm->info() << "t = Ax";
    algebra.mult(A, x, tmp);
    tm->info() << "r = t";
    algebra.copy(tmp, r);
    tm->info() << "r -= b";
    algebra.axpy(-1., b, r);
  }

  auto norm = algebra.norm2(r);

  tm->info() << " => ||r|| = " << norm;

#ifdef ALIEN_USE_PETSC
  if (arguments.count("dump-on-screen")) {
    tm->info() << "dump solution after solve";
    Alien::dump(x);
  }
#endif

  tm->info() << "* r = || x - xe ||";

  {
    tm->info() << "r = x";
    algebra.copy(x, r);
    tm->info() << "r -= xe";
    algebra.axpy(-1., xe, r);
  }

  tm->info() << " => ||r|| = " << norm;

  // double tol = arguments["tol"].as<double>();
  if (norm > 1e-3)
    tm->fatal() << "||r|| too big";

  tm->info() << " ";
  tm->info() << "... example finished !!!";
}


// Méthode de construction de la matrice
extern void buildMatrix(
    Alien::BlockMatrix& A, std::string const& filename, std::string const& format);

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
        boost::program_options::value<int>()->default_value(100), "size of problem")(
        "block-size", boost::program_options::value<int>()->default_value(5),
        "size of block")("file-name",
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

    int size = arguments["size"].as<int>();
    int block_size = arguments["block-size"].as<int>();
    std::string filename = arguments["file-name"].as<std::string>();
    std::string format = arguments["format"].as<std::string>();

    tm->info() << "Example Alien :";
    tm->info()
        << "Use of blocks builder (RefSemanticMVHandlers API) for Laplacian problem";
    tm->info() << " => solving linear system Ax = b";
    tm->info() << " * problem size = " << size;
    tm->info() << " * block size = " << block_size;
    tm->info() << " ";
    tm->info() << "Start example...";
    tm->info() << " ";

    Alien::setTraceMng(tm);

    Alien::setVerbosityLevel(Alien::Verbosity::Debug);

    const Alien::Block block(block_size);
    Alien::MatrixDistribution dist(size, size, pm);

    Alien::BlockMatrix A(size, size, block, pm);

    size = A.rowSpace().size();
    block_size = A.block().size();

    tm->info() << "=> Matrix Distribution : " << A.distribution();

    buildMatrix(A, filename, format);

    if (arguments.count("dump-on-screen")) {
#ifdef ALIEN_USE_PETSC
      tm->info() << "dump A on screen";
      Alien::dump(A);
#endif // ALIEN_USE_PETSC
    }

    tm->info() << "* xe = 1";

    Alien::BlockVector xe = Alien::ones(size, block, pm);

    tm->info() << "=> Vector Distribution : " << xe.distribution();

    if (arguments.count("dump-on-screen")) {
#ifdef ALIEN_USE_PETSC
      tm->info() << "dump exact solution xe";
      Alien::dump(xe);
#endif // ALIEN_USE_PETSC
    }

    tm->info() << "* b = A * xe";

    Alien::BlockVector b(size, block, pm);


    const auto solver_package = arguments["solver-package"].as<std::string>();
#ifdef ALIEN_USE_PETSC
    if(solver_package == "petsc") {
      Alien::PETScInternalLinearSolver::initializeLibrary() ;
      Alien::PETScLinearAlgebra algebra;

      test(tm, format, pm, arguments, algebra, size, block, A, xe, b);
    }
#endif // ALIEN_USE_PETSC

#ifdef ALIEN_USE_HYPRE
    if(solver_package == "hypre") {
      //Alien::HypreLinearAlgebra algebra;
      Alien::SimpleCSRLinearAlgebra algebra; // Hypre Algebra not complete

      test(tm, format, pm, arguments, algebra, size, block, A, xe, b);
    }
#endif

#ifdef ALIEN_USE_IFPSOLVER
    if(solver_package == "ifpsolver") {
      Alien::SimpleCSRLinearAlgebra algebra;

      test(tm, format, pm, arguments, algebra, size, block, A, xe, b);
    }
#endif

#ifdef ALIEN_USE_MCGSOLVER
    if(solver_package == "mcgsolver") {
      Alien::SimpleCSRLinearAlgebra algebra;

      test(tm, format, pm, arguments, algebra, size, block, A, xe, b);
    }
#endif

    return 0;
  });
}
