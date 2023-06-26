#include <gtest/gtest.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/data/Space.h>
#include <alien/expression/solver/IEigenSolver.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/core/backend/EigenSolver.h>
#include <alien/core/backend/EigenSolverT.h>
#include <alien/core/backend/IInternalEigenSolverT.h>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>

#ifdef ALIEN_USE_HARTS
#include "HARTS/HARTS.h"
#endif

#ifdef ALIEN_USE_HTSSOLVER
#include "HARTSSolver/HTS.h"
#include "HARTSSolver/MatrixVector/CSR/CSRProfileImpT.h"
#include "HARTSSolver/MatrixVector/CSR/CSRMatrixImpT.h"

#include <ALIEN/axl/HTSEigenSolver_IOptions.h>
#include <ALIEN/Kernels/hts/eigen_solver/HTSEigenOptionTypes.h>
#include <ALIEN/Kernels/hts/eigen_solver/HTSInternalEigenSolver.h>

#include <ALIEN/Kernels/hts/eigen_solver/arcane/HTSEigenSolver.h>
#include <ALIEN/axl/HTSEigenSolver_IOptions.h>
#include <ALIEN/axl/HTSEigenSolver_StrongOptions.h>
#endif

#ifdef ALIEN_USE_SLEPC
#include <ALIEN/Kernels/PETSc/data_structure/PETScVector.h>
#include <ALIEN/Kernels/PETSc/data_structure/PETScMatrix.h>
#include <ALIEN/Kernels/PETSc/PETScBackEnd.h>

#include <ALIEN/axl/SLEPcEigenSolver_IOptions.h>
#include <ALIEN/Kernels/PETSc/eigen_solver/SLEPcEigenOptionTypes.h>
#include <ALIEN/Kernels/PETSc/eigen_solver/SLEPcInternalEigenSolver.h>

#include <ALIEN/Kernels/PETSc/eigen_solver/arcane/SLEPcEigenSolver.h>
#include <ALIEN/axl/SLEPcEigenSolver_IOptions.h>
#include <ALIEN/axl/SLEPcEigenSolver_StrongOptions.h>
#endif

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
extern Arccore::ITraceMng* traceMng();
} // namespace Environment

#ifdef ALIEN_USE_HTSSOLVER
// Tests the default c'tor.
TEST(TestEigenSolver, HTSEigenSolverSmallestValue)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = Environment::traceMng();
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VectorDistribution vdist(s, Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Matrix B(mdist); // A.setName("A") ;

  Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(3 * local_size);
    builder.allocate();
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }

  using namespace HTSEigenSolverOptionsNames;
  std::shared_ptr<IOptionsHTSEigenSolver> options(
      new StrongOptionsHTSEigenSolver{ _output = 1, _maxIterationNum = 1000, _tol = 1e-6,
          _evType = 2, _nev = 5, _evOrder = 0 });
  auto* solver = new Alien::HTSEigenSolver(Environment::parallelMng(), options);
  //
  solver->init();
  EigenProblemT<Alien::BackEnd::tag::simplecsr, std::vector<double>> problem(A);
  solver->solve(problem);
  const Alien::IEigenSolver::Status& status = solver->getStatus();
  if (status.m_succeeded) {
    std::cout << "Solver succeed        " << status.m_succeeded << std::endl;
    std::cout << "             residual " << status.m_residual << std::endl;
    std::cout << "             nb iters " << status.m_iteration_count << std::endl;
    std::cout << "             error    " << status.m_error << std::endl;
    std::cout << "             nconv    " << status.m_nconv << std::endl;
    auto& r_values = problem.getRealEigenValues();
    std::cout << "Real Eigen Values" << std::endl;
    for (Integer i = 0; i < status.m_nconv; ++i)
      std::cout << "(" << i << "," << r_values[i] << ") " << std::endl;
  }
  delete solver;
}

TEST(TestEigenSolver, HTSGenEigenSolverSmallestValue)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = Environment::traceMng();
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VectorDistribution vdist(s, Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Matrix B(mdist); // A.setName("A") ;

  Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(3 * local_size);
    builder.allocate();
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }

  {
    Alien::DirectMatrixBuilder builder(B, tag);
    builder.reserve(local_size);
    builder.allocate();
    for (Integer i = 0; i < 5; ++i) {
      Integer row = offset + i;
      builder(row, row) = 0.1 * row;
    }
  }

  using namespace HTSEigenSolverOptionsNames;
  std::shared_ptr<IOptionsHTSEigenSolver> options(
      new StrongOptionsHTSEigenSolver{ _output = 1, _maxIterationNum = 1000, _tol = 1e-6,
          _evType = 2, _nev = 5, _evOrder = 0 });
  auto* solver = new Alien::HTSEigenSolver(Environment::parallelMng(), options);
  //
  solver->init();
  GeneralizedEigenProblemT<Alien::BackEnd::tag::simplecsr, std::vector<double>> problem(
      A, B);
  solver->solve(problem);
  const Alien::IEigenSolver::Status& status = solver->getStatus();
  if (status.m_succeeded) {
    std::cout << "Solver succeed        " << status.m_succeeded << std::endl;
    std::cout << "             residual " << status.m_residual << std::endl;
    std::cout << "             nb iters " << status.m_iteration_count << std::endl;
    std::cout << "             error    " << status.m_error << std::endl;
    std::cout << "             nconv    " << status.m_nconv << std::endl;
    auto& r_values = problem.getRealEigenValues();
    std::cout << "Real Eigen Values" << std::endl;
    for (Integer i = 0; i < status.m_nconv; ++i)
      std::cout << "(" << i << "," << r_values[i] << ") " << std::endl;
  }
  delete solver;
}
#endif

#ifdef ALIEN_USE_SLEPC
// Tests the default c'tor.
TEST(TestEigenSolver, SLEPcEigenSolverSmallestValue)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = Environment::traceMng();
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VectorDistribution vdist(s, Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Matrix B(mdist); // A.setName("A") ;

  Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(3 * local_size);
    builder.allocate();
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }

  using namespace SLEPcEigenSolverOptionsNames;
  std::shared_ptr<IOptionsSLEPcEigenSolver> options(
      new StrongOptionsSLEPcEigenSolver{ _output = 1, _maxIterationNum = 1000,
          _tol = 1e-6, _evType = 0, _nev = 5, _evOrder = 0 });
  auto* solver = new Alien::SLEPcEigenSolver(Environment::parallelMng(), options);
  //
  solver->init();
  EigenProblemT<Alien::BackEnd::tag::petsc, std::vector<double>> problem(A);
  solver->solve(problem);
  const Alien::IEigenSolver::Status& status = solver->getStatus();
  if (status.m_succeeded) {
    std::cout << "Solver succeed        " << status.m_succeeded << std::endl;
    std::cout << "             residual " << status.m_residual << std::endl;
    std::cout << "             nb iters " << status.m_iteration_count << std::endl;
    std::cout << "             error    " << status.m_error << std::endl;
    std::cout << "             nconv    " << status.m_nconv << std::endl;
    auto& r_values = problem.getRealEigenValues();
    std::cout << "Real Eigen Values" << std::endl;
    for (Integer i = 0; i < status.m_nconv; ++i)
      std::cout << "(" << i << "," << r_values[i] << ") " << std::endl;
  }
  delete solver;
}

TEST(TestEigenSolver, SLEPcGenEigenSolverSmallestValue)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = Environment::traceMng();
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VectorDistribution vdist(s, Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Matrix B(mdist); // A.setName("A") ;

  Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(3 * local_size);
    builder.allocate();
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }

  {
    Alien::DirectMatrixBuilder builder(B, tag);
    builder.reserve(local_size);
    builder.allocate();
    for (Integer i = 0; i < 5; ++i) {
      Integer row = offset + i;
      builder(row, row) = 0.1 * row;
    }
  }

  using namespace SLEPcEigenSolverOptionsNames;
  std::shared_ptr<IOptionsSLEPcEigenSolver> options(
      new StrongOptionsSLEPcEigenSolver{ _output = 1, _maxIterationNum = 1000,
          _tol = 1e-6, _evType = 0, _nev = 5, _evOrder = 0 });
  auto* solver = new Alien::SLEPcEigenSolver(Environment::parallelMng(), options);
  //
  solver->init();
  GeneralizedEigenProblemT<Alien::BackEnd::tag::petsc, std::vector<double>> problem(A, B);
  solver->solve(problem);
  const Alien::IEigenSolver::Status& status = solver->getStatus();
  if (status.m_succeeded) {
    std::cout << "Solver succeed        " << status.m_succeeded << std::endl;
    std::cout << "             residual " << status.m_residual << std::endl;
    std::cout << "             nb iters " << status.m_iteration_count << std::endl;
    std::cout << "             error    " << status.m_error << std::endl;
    std::cout << "             nconv    " << status.m_nconv << std::endl;
    auto& r_values = problem.getRealEigenValues();
    std::cout << "Real Eigen Values" << std::endl;
    for (Integer i = 0; i < status.m_nconv; ++i)
      std::cout << "(" << i << "," << r_values[i] << ") " << std::endl;
  }
  delete solver;
}
#endif
