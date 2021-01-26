#include <Tests/Options.h>

#include <alien/Alien.h>
#include <alien/AlienExternalPackages.h>
#include <alien/ref/AlienRefSemantic.h>

#include <Tests/Solver.h>

// Méthode de construction de la matrice
extern void buildMatrix(Alien::VBlockMatrix& A);

int
main(int argc, char** argv)
{
  return Environment::execute(argc, argv, [&] {

    auto* pm = Environment::parallelMng();
    auto* tm = Environment::traceMng();

    // Options pour ce test
    boost::program_options::options_description options;
    options.add_options()("help", "print usage")(
        "dump-on-screen", "dump algebraic objects on screen")("size",
        boost::program_options::value<int>()->default_value(100), "size of problem");

    // On recupere les options (+ celles de la configuration des solveurs)
    auto arguments = Environment::options(argc, argv, options);

    if (arguments.count("help")) {
      tm->info() << "Usage :\n" << options;
      return 1;
    }

    int size = arguments["size"].as<int>();

    tm->info() << "Example Alien :";
    tm->info()
        << "Use of vblocks builder (RefSemanticMVHandlers API) for Laplacian problem";
    tm->info() << " => solving linear system Ax = b";
    tm->info() << " * problem size = " << size;
    tm->info() << " ";
    tm->info() << "Start example...";
    tm->info() << " ";

    Alien::setTraceMng(tm);

    Alien::setVerbosityLevel(Alien::Verbosity::Debug);

    Alien::VBlock::ValuePerBlock v;
    for (Alien::Integer i = 0; i < size; ++i)
      v[i] = 3;

    const Alien::VBlock block(std::move(v));
    Alien::MatrixDistribution dist(size, size, pm);

    Alien::VBlockMatrix A(block, dist);

    tm->info() << "=> Matrix Distribution : " << A.distribution();

    buildMatrix(A);

    if (arguments.count("dump-on-screen")) {
#ifdef ALIEN_USE_PETSC
      tm->info() << "dump A on screen";
      Alien::dump(A, Alien::AsciiDumper::Style::eSequentialVariableBlockSizeStype);
#endif // ALIEN_USE_PETSC
    }

    tm->info() << "* xe = 1";

    Alien::VBlockVector xe = Alien::ones(size, block, pm);

    tm->info() << "=> Vector Distribution : " << xe.distribution();

    if (arguments.count("dump-on-screen")) {
#ifdef ALIEN_USE_PETSC
      tm->info() << "dump exact solution xe";
      Alien::dump(xe);
#endif // ALIEN_USE_PETC
    }

    tm->info() << "* b = A * xe";

    Alien::VBlockVector b(size, block, pm);

    Alien::SimpleCSRLinearAlgebra algebra;

    algebra.mult(A, xe, b);

    if (arguments.count("dump-on-screen")) {
#ifdef ALIEN_USE_PETSC
      tm->info() << "dump b";
      Alien::dump(b);
#endif // ALIEN_USE_PETSC
    }

    tm->info() << " ** [solver package=" << arguments["solver-package"].as<std::string>()
               << "]";

    Alien::VBlockVector x(size, block, pm);

    tm->info() << "* x = A^-1 b";

    auto solver = Environment::createSolver(arguments);
    solver->init();
    solver->solve(A, b, x);

    tm->info() << "* r = Ax - b";

    Alien::VBlockVector r(size, block, pm);

    {
      Alien::VBlockVector tmp(size, block, pm);
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
#ifdef ALIEN_USE_PETSC
      tm->info() << "dump solution after solve";
      Alien::dump(x);
#endif // ALIEN_USE_PETSC
    }

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

    return 0;
  });
}
