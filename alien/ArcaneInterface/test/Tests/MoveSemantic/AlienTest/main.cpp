#include <string>
#include <map>
#include <time.h>
#include <vector>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#if defined ALIEN_USE_MTL4 || defined ALIEN_USE_PETSC || defined ALIEN_USE_HYPRE
#include <alien/AlienExternalPackages.h>
#endif
#if defined ALIEN_USE_IFPSOLVER || defined ALIEN_USE_MCGSOLVER
#include <alien/AlienIFPENSolvers.h>
#endif
#include <alien/move/AlienMoveSemantic.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <Tests/Environment.h>
#include <Tests/Solver.h>

int
main(int argc, char** argv)
{
  return Environment::execute(argc, argv, [&] {

    using namespace boost::program_options;
    options_description desc;
    desc.add_options()("help", "produce help")("file",
        value<std::string>()->default_value("SystemFile.txt"),
        "system imput file")("nrows", value<int>()->default_value(0), "nrow")("nx",
        value<int>()->default_value(10), "nx")("ny", value<int>()->default_value(10),
        "ny")("nbth", value<int>()->default_value(1), "number of threads")(
        "solver-package", value<std::string>()->default_value("petsc"),
        "solver package name")("solver", value<std::string>()->default_value("bicgs"),
        "solver algo name")("precond", value<std::string>()->default_value("none"),
        "preconditioner id diag ilu ddml poly")(
        "max-iter", value<int>()->default_value(1000), "max iterations")(
        "tol", value<double>()->default_value(1.e-10), "solver tolerance")(
        "niter", value<int>()->default_value(1), "nb of tests for perf measure")(
        "sym", value<int>()->default_value(1), "0->nsym, 1->sym")("kernel",
        value<std::string>()->default_value("mcgkernel"),
        "mcgsolver kernel name")("builder", value<int>()->default_value(0),
        "matrix builder type 0->direct 1->profile 2->stream")(
        "output-level", value<int>()->default_value(0), "output level");
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }

    std::string filename = vm["file"].as<std::string>();
    int nrows = vm["nrows"].as<int>();
    int nx = vm["nx"].as<int>();
    int ny = vm["ny"].as<int>();
    //  int nb_threads = vm["nbth"].as<int>();
    //  bool sym = vm["sym"].as<int>() == 1;
    int builder_type = vm["builder"].as<int>();

    auto* trace_mng = Environment::traceMng();
    auto* parallel_mng = Environment::parallelMng();

    const int comm_rank = parallel_mng->commRank();
    const int comm_size = parallel_mng->commSize();

    Alien::setTraceMng(trace_mng);

    nrows = nx * ny;

    Alien::VectorDistribution vdist(nrows, parallel_mng);
    Alien::MatrixDistribution mdist(nrows, nrows, parallel_mng);

    Alien::Space space(nrows, "TestSpace");

    Alien::Move::VectorData vectorB(space, vdist);
    Alien::Move::VectorData vectorX(space, vdist);

    Alien::Move::MatrixData matrixA(space, mdist);

    int bd = 2;
    double fii = 1;
    double fij = 1;
    double diag = fii + 2 * bd * fij;
    double offdiag = fij;
    int offset = vdist.offset();
    int local_size = vdist.localSize();
    switch (builder_type) {
    case 0: {
      Alien::Move::DirectMatrixBuilder builder(
          std::move(matrixA), Alien::DirectMatrixOptions::eResetValues);
      builder.reserve(5);
      builder.allocate();
      for (int irow = offset; irow < offset + local_size; ++irow) {
        builder.setData(irow, irow, diag);
        for (int j = 1; j < bd; ++j) {
          if (irow - j >= 0)
            builder.setData(irow, irow - j, -offdiag / j);
          if (irow + j < nrows)
            builder.setData(irow, irow + j, -offdiag / j);
        }
      }
      builder.finalize();
      matrixA = builder.release();
    } break;
    case 1:
    default: {
      // DEFINE MATRIX PROFILE
      Alien::Move::MatrixProfiler profile(std::move(matrixA));
      for (int irow = offset; irow < offset + local_size; ++irow) {
        profile.addMatrixEntry(irow, irow);
        for (int j = 1; j < bd; ++j) {
          if (irow - j >= 0)
            profile.addMatrixEntry(irow, irow - j);
          if (irow + j < nrows)
            profile.addMatrixEntry(irow, irow + j);
        }
      }
      profile.allocate();
      matrixA = profile.release();

      // FILL MATRIX
      Alien::Move::ProfiledMatrixBuilder matrix(
          std::move(matrixA), Alien::ProfiledMatrixOptions::eResetValues);
      for (int irow = offset; irow < offset + local_size; ++irow) {
        matrix(irow, irow) = diag;
        for (int j = 1; j < bd; ++j) {
          if (irow - j >= 0)
            matrix(irow, irow - j) = -offdiag / j;
          if (irow + j < nrows)
            matrix(irow, irow + j) = -offdiag / j;
        }
      }
      matrix.finalize();
      matrixA = matrix.release();
    } break;
    }

    {
      Alien::Move::LocalVectorWriter v(std::move(vectorX));
      for (int i = 0; i < vdist.localSize(); ++i)
        v[i] = 1;
      vectorX = v.release();
    }
    double normx_ref = 0;
    if (!(builder_type == 0) || comm_size == 1) // le direct builder ne permet pas de
                                                // faire des produits matrix vector en
                                                // parallel
    {
      Alien::SimpleCSRLinearAlgebra alg;
      alg.mult(matrixA, vectorX, vectorB);
      double normb = alg.norm2(vectorB);
      normx_ref = alg.norm2(vectorX);
      if (comm_rank == 0) {
        std::cout << "Norme de X : " << normx_ref << std::endl;
        std::cout << "Norme de B = A * x : " << normb << std::endl;
        std::cout << "CHECK ERROR " << std::abs(normb - 30.2324) << std::endl;
      }
      assert(std::abs(normb - 30.2324) < 1.e-4);
    } else {
      Alien::Move::LocalVectorWriter v(std::move(vectorB));
      for (int i = 0; i < vdist.localSize(); ++i)
        v[i] = 1;
      vectorB = v.release();
    }

    {
      Alien::Move::LocalVectorWriter v(std::move(vectorX));
      for (int i = 0; i < vdist.localSize(); ++i)
        v[i] = 0;
      vectorX = v.release();
    }

    std::shared_ptr<Alien::ILinearSolver> solver = Environment::createSolver(vm);
    if (solver.get()) {
      solver->init();
      solver->solve(matrixA, vectorB, vectorX);
      const auto& status = solver->getStatus();
      if (status.succeeded) {
        if (comm_rank == 0) {
          std::cout << "Solver succeed   " << status.succeeded << std::endl;
          std::cout << "             residual " << status.residual << std::endl;
          std::cout << "             nb iters  " << status.iteration_count << std::endl;
          std::cout << "             error      " << status.error << std::endl;
        }

        if (!(builder_type == 0) || comm_size == 1) // le direct builder ne permet pas de
                                                    // faire des produits matrix vector en
                                                    // parallel
        {
          Alien::SimpleCSRLinearAlgebra alg;
          double normx = alg.norm2(vectorX);
          if (comm_rank == 0) {
            std::cout << "Norme de X : " << normx << std::endl;
          }

          Alien::Move::LocalVectorWriter v(std::move(vectorX));
          for (int i = 0; i < vdist.localSize(); ++i)
            v[i] -= 1.;
          vectorX = v.release();

          normx = alg.norm2(vectorX);
          double err = normx / normx_ref;
          if (comm_rank == 0) {
            std::cout << "Relative Error : " << err << std::endl;
          }
          assert(err < 1.e-10);
        }
      } else {
        if (comm_rank == 0) {
          std::cout << "Solver failed      " << status.succeeded << std::endl;
          std::cout << "             error      " << status.error << std::endl;
        }
      }
    } else
      std::cout << "No solver provided\n";

    return 0;
  });
}
