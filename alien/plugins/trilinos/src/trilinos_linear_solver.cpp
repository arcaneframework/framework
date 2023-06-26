/*
 * Copyright 2021 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trilinos_linear_solver.h"

#include <Ifpack2_Factory.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosSolverFactory_Tpetra.hpp>
#include <Tpetra_Operator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

namespace Alien
{
// Compile TrilinosLinearSolver.
template class ALIEN_TRILINOS_EXPORT LinearSolver<BackEnd::tag::trilinos>;
} // namespace Alien

namespace Alien::Trilinos
{

bool InternalLinearSolver::solve(const Matrix& A, const Vector& b, Vector& x)
{
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Create the linear problem instance.
  Belos::LinearProblem<SC, MV, OP> problem(A.internal(), x.internal(), b.internal());

  // preconditioners.
  switch (m_options.preconditioner()) {
  case OptionTypes::MueLu: {
    Teuchos::ParameterList paramList;
    paramList.set("verbosity", "low");
    paramList.set("max levels", 3);
    paramList.set("coarse: max size", 10);
    paramList.set("multigrid algorithm", "sa");
    auto prec = MueLu::CreateTpetraPreconditioner(static_cast<Teuchos::RCP<OP>>(A.internal()), paramList);
    problem.setLeftPrec(prec);
  } break;
  case OptionTypes::Relaxation: {
    auto M = Ifpack2::Factory::create<row_matrix_type>("RELAXATION", A.internal());
    if (M.is_null()) {
      std::cerr << "Failed to create Ifpack2 preconditioner !" << std::endl;
      return -1;
    }
    M->initialize();
    M->compute();

    problem.setLeftPrec(M);
  } break;
  case OptionTypes::NoPC:
    break;
  default:
    alien_fatal([this] { cout() << "Undefined IFPACK2 preconditioner option"; });
    break;
  }

  // Set the problem
  if (!problem.setProblem()) {
    std::cout << std::endl
              << "ERROR:  Belos::LinearProblem failed to set up correctly !" << std::endl;
    return -1;
  }

  // Create the solver
  ParameterList belosList;
  belosList.set("Maximum Iterations", m_options.numIterationsMax());
  belosList.set("Convergence Tolerance", m_options.stopCriteriaValue());

  std::unique_ptr<Belos::SolverManager<SC, MV, OP>> solver;

  // allocate solver
  switch (m_options.solver()) {
  case OptionTypes::CG:
    solver = std::make_unique<Belos::BlockCGSolMgr<SC, MV, OP>>(rcpFromRef(problem), rcpFromRef(belosList));
    break;
  case OptionTypes::GMRES:
    solver = std::make_unique<Belos::BlockGmresSolMgr<SC, MV, OP>>(rcpFromRef(problem), rcpFromRef(belosList));
    break;
  case OptionTypes::BICGSTAB:
    solver = std::make_unique<Belos::BiCGStabSolMgr<SC, MV, OP>>(rcpFromRef(problem), rcpFromRef(belosList));
    break;
  default:
    alien_fatal([this] {
      cout() << "Undefined solver option";
    });
    break;
  }

  // Init timer
  double tic = MPI_Wtime();
  Belos::ReturnType ret = solver->solve();
  double toc = MPI_Wtime();
  double sec = toc - tic;

  // Check
  if (ret == Belos::Converged) {
    // Get solver and timing infos
    const int numIters = solver->getNumIters();
    double itPerSec = numIters / sec;

    // Print
    if (int rank = A.internal()->getComm()->getRank(); rank == 0) {
      std::cout << "Belos Solver has converged." << std::endl;
      kokkos_node_verbose();
      std::cout << "numIters : " << numIters << std::endl;
      std::cout << "achieved tol : " << solver->achievedTol() << std::endl;
      std::cout << "Execution time [s]: " << sec << std::endl;
      std::cout << "Iterations per second : " << itPerSec << std::endl;
    }

    // update solver infos
    /*m_status.residual = residual_norm;*/
    m_status.iteration_count = numIters;
    m_status.succeeded = true;
    m_total_iter_num += m_status.iteration_count;
    ++m_solve_num;
    m_total_solve_time += sec;
  }
  else if (ret == Belos::Unconverged) {
    m_status.succeeded = false;
    std::cout << "Belos Solver did not converge !" << std::endl;
  }

  return m_status.succeeded;
}

const Alien::SolverStatus&
InternalLinearSolver::getStatus() const
{
  return m_status;
}

ALIEN_TRILINOS_EXPORT
std::shared_ptr<ILinearAlgebra>
InternalLinearSolver::algebra() const
{
  return std::make_shared<LinearAlgebra>();
}

ALIEN_TRILINOS_EXPORT
IInternalLinearSolver<Matrix, Vector>*
InternalLinearSolverFactory(const Options& options)
{
  return new InternalLinearSolver(options);
}

ALIEN_TRILINOS_EXPORT
IInternalLinearSolver<Matrix, Vector>*
InternalLinearSolverFactory()
{
  return new InternalLinearSolver();
}
} // namespace Alien::Trilinos
