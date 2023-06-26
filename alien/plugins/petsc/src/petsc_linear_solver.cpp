/*
 * Copyright 2020 IFPEN-CEA
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

#include "matrix.h"
#include "vector.h"

#include <petscksp.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/LinearSolverT.h>

#include <alien/petsc/backend.h>
#include <alien/petsc/options.h>
#include <alien/petsc/export.h>

#include "petsc_instance.h"

namespace Alien
{
// Compile PetscLinearSolver.
template class ALIEN_PETSC_EXPORT LinearSolver<BackEnd::tag::petsc>;

} // namespace Alien

namespace Alien::PETSc
{
class InternalLinearSolver
: public IInternalLinearSolver<Matrix, Vector>
, public ObjectWithTrace
{
 public:
  typedef SolverStatus Status;

  InternalLinearSolver();

  explicit InternalLinearSolver(const Options& options);

  ~InternalLinearSolver() override = default;

 public:
  // Nothing to do
  void updateParallelMng(ALIEN_UNUSED_PARAM
                         Arccore::MessagePassing::IMessagePassingMng* pm) override {}

  bool solve(const Matrix& A, const Vector& b, Vector& x) override;

  bool hasParallelSupport() const override { return true; }

  //! Etat du solveur
  const Status& getStatus() const override;

  const SolverStat& getSolverStat() const override { return m_stat; }

  SolverStat& getSolverStat() override { return m_stat; }

  std::shared_ptr<ILinearAlgebra> algebra() const override;

 private:
  Status m_status;

  Arccore::Real m_init_time{ 0.0 };
  Arccore::Real m_total_solve_time{ 0.0 };
  Arccore::Integer m_solve_num{ 0 };
  Arccore::Integer m_total_iter_num{ 0 };

  SolverStat m_stat;
  Options m_options;

 private:
  void checkError(const Arccore::String& msg, int ierr,
                  int skipError = 0) const;
};

InternalLinearSolver::InternalLinearSolver()
: m_status()
, m_stat()
, m_options()
{
  petsc_init_if_needed();
}

InternalLinearSolver::InternalLinearSolver(const Options& options)
: m_status()
, m_stat()
, m_options(options)
{
  petsc_init_if_needed();
}

void InternalLinearSolver::checkError(const Arccore::String& msg, int ierr, int skipError) const
{
  if (ierr != 0 and (ierr & ~skipError) != 0) {
    alien_fatal([&] {
      cout() << msg << " failed. [code=" << ierr << "]";
      CHKERRV(ierr);
    });
  }
}

bool InternalLinearSolver::solve(const Matrix& A, const Vector& b, Vector& x)
{
  auto tsolve = MPI_Wtime();

  ALIEN_UNUSED_PARAM int output_level = m_options.verbose() ? 1 : 0;

  // failback if no MPI comm already defined.
  MPI_Comm comm = MPI_COMM_WORLD;
  auto* mpi_comm_mng = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(A.distribution().parallelMng());
  if (mpi_comm_mng)
    comm = *(mpi_comm_mng->getMPIComm());

  // solver's choice
  // Liste à compléter (dans options.h), on met lesquels ?
  std::string solver_name = "undefined";
  switch (m_options.solver()) {
  case OptionTypes::GMRES:
    solver_name = "gmres";
    break;
  case OptionTypes::CG:
    solver_name = "cg";
    break;
  case OptionTypes::BiCG:
    solver_name = "bicg";
    break;
  case OptionTypes::BiCGstab:
    solver_name = "bcgs";
    break;
  default:
    alien_fatal([&] {
      cout() << "Undefined solver option";
    });
    break;
  }

  // preconditioner's choice
  // Liste à compléter (dans options.h), on met lesquels ?
  std::string precond_name = "undefined";
  switch (m_options.preconditioner()) {
  case OptionTypes::Jacobi:
    precond_name = "jacobi";
    break;
  case OptionTypes::NoPC:
    precond_name = "none";
    break;
  default:
    alien_fatal([&] { cout() << "Undefined Petsc preconditioner option"; });
    break;
  }

  // Get options and configure solver + preconditioner
  KSP solver;
  checkError("PETSc create solver", KSPCreate(comm, &solver));
  checkError("PETSc set solver type",
             KSPSetType(solver, solver_name.c_str()));
  checkError("PETSc set Operators",
             KSPSetOperators(solver, A.internal(), A.internal()));
  // Here the matrix that defines the linear system also serves as the
  // preconditioning matrix

  PC preconditioner;
  checkError("PETSc get the preconditioner",
             KSPGetPC(solver, &preconditioner));
  int max_it = m_options.numIterationsMax();
  double rtol = m_options.stopCriteriaValue();
  checkError("PETSc set the preconditioner",
             PCSetType(preconditioner,
                       precond_name.c_str())); // petsc prend un char *
  checkError(
  "PETSc set tolerances",
  KSPSetTolerances(solver, rtol, PETSC_DEFAULT, PETSC_DEFAULT, max_it));

  // solve
  m_status.succeeded = (KSPSolve(solver, b.internal(), x.internal()) == 0);

  // get nb iterations + final residual
  checkError("PETSc get iteration number", KSPGetIterationNumber(solver, &m_status.iteration_count));
  checkError("PETSc get residual norm", KSPGetResidualNorm(solver, &m_status.residual));

  // destroy solver + pc
  checkError("PETSc destroy solver context", KSPDestroy(&solver)); // includes a call to PCDestroy

  // update the counters
  ++m_solve_num;
  m_total_iter_num += m_status.iteration_count;
  tsolve = MPI_Wtime() - tsolve;
  m_total_solve_time += tsolve;

  return m_status.succeeded;
}

const Alien::SolverStatus&
InternalLinearSolver::getStatus() const
{
  return m_status;
}

ALIEN_PETSC_EXPORT
std::shared_ptr<ILinearAlgebra>
InternalLinearSolver::algebra() const
{
  return std::make_shared<LinearAlgebra>();
}

ALIEN_PETSC_EXPORT
IInternalLinearSolver<Matrix, Vector>*
InternalLinearSolverFactory(const Options& options)
{
  return new InternalLinearSolver(options);
}

ALIEN_PETSC_EXPORT
IInternalLinearSolver<Matrix, Vector>*
InternalLinearSolverFactory()
{
  return new InternalLinearSolver();
}
} // namespace Alien::PETSc
