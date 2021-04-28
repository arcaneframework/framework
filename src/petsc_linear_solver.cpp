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

#include <boost/timer.hpp>

#include <petscksp.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/LinearSolverT.h>

#include <alien/petsc/backend.h>
#include <alien/petsc/options.h>
#include <alien/petsc/export.h>

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

    InternalLinearSolver(const Options& options);

    virtual ~InternalLinearSolver() {}

   public:
    // Nothing to do
    void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm) {}

    bool solve(const Matrix& A, const Vector& b, Vector& x);

    bool hasParallelSupport() const { return true; }

    //! Etat du solveur
    const Status& getStatus() const;

    const SolverStat& getSolverStat() const { return m_stat; }

    std::shared_ptr<ILinearAlgebra> algebra() const;

   private:
    Status m_status;

    Arccore::Real m_init_time;
    Arccore::Real m_total_solve_time;
    Arccore::Integer m_solve_num;
    Arccore::Integer m_total_iter_num;

    SolverStat m_stat;
    Options m_options;

   private:
    void checkError(const Arccore::String& msg, int ierr, int skipError = 0) const;
  };

  InternalLinearSolver::InternalLinearSolver()
  {
    boost::timer tinit;
    m_init_time += tinit.elapsed();
  }

  InternalLinearSolver::InternalLinearSolver(const Options& options)
  : m_options(options)
  {
    boost::timer tinit;
    m_init_time += tinit.elapsed();
  }

  void InternalLinearSolver::checkError(const Arccore::String& msg, int ierr, int skipError) const
  {/*
    if (ierr != 0 and (ierr & ~skipError) != 0) {
      char hypre_error_msg[256];
      HYPRE_DescribeError(ierr, hypre_error_msg);
      alien_fatal([&] {
        cout() << msg << " failed : " << hypre_error_msg << "[code=" << ierr << "]";
      });
    }*/
  }

  bool InternalLinearSolver::solve(const Matrix& A, const Vector& b, Vector& x)
  {
    
    auto ij_matrix = A.internal();
    auto bij_vector = b.internal();
    auto xij_vector = x.internal();

    // Macro "pratique" en attendant de trouver mieux
    boost::timer tsolve;

    int output_level = m_options.verbose() ? 1 : 0;

    // Je suppose qu'il faut utiliser le communicateur d'Alien, comme pour Hypre ? 
    MPI_Comm comm = MPI_COMM_WORLD;
    auto* mpi_comm_mng = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(A.distribution().parallelMng());
    if (mpi_comm_mng)
      comm = *(mpi_comm_mng->getMPIComm());

    /* types necessaires, clean later */
    PetscErrorCode ierr;    

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
      alien_fatal([&] {
        cout() << "Undefined Petsc preconditioner option";
      });
      break;
    }
    
    // Get options and configure solver + preconditioner
    KSP solver;    
    ierr = KSPCreate(comm,&solver);CHKERRQ(ierr);
    ierr = KSPSetType(solver,solver_name.c_str());    
    ierr = KSPSetOperators(solver,A.internal(),A.internal());CHKERRQ(ierr); //Here the matrix that defines the linear system also serves as the preconditioning matrix      
    
    PC preconditioner;
    ierr = KSPGetPC(solver,&preconditioner);CHKERRQ(ierr);        
    int max_it = m_options.numIterationsMax();
    double rtol = m_options.stopCriteriaValue();
    ierr = PCSetType(preconditioner,precond_name.c_str());CHKERRQ(ierr); // petsc prend un char *
    ierr = KSPSetTolerances(solver,rtol,PETSC_DEFAULT,PETSC_DEFAULT,max_it);CHKERRQ(ierr);   

    // solve
    m_status.succeeded = (KSPSolve(solver,b.internal(),x.internal()) == 0);

    // get nb iterations + final residual
    KSPGetIterationNumber(solver, &m_status.iteration_count);
    KSPGetResidualNorm(solver,&m_status.residual);

    // pour info, à virer
    std::cout<<"================ solver " << solver_name << std::endl;
    std::cout<<"================ preconditioner " << precond_name << std::endl;   
    std::cout<<"================ nb iterations " << m_status.iteration_count << std::endl;
    std::cout<<"================ Final residual norm " << m_status.residual << std::endl;   

    // destroy solver + pc
    KSPDestroy(&solver); // includes a call to PCDestroy

    // update the counters
    ++m_solve_num;
    m_total_iter_num += m_status.iteration_count;
    m_total_solve_time += tsolve.elapsed();
    
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
