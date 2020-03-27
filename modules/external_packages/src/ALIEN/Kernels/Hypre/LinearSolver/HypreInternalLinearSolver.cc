#include "HypreInternalLinearSolver.h"

#include <memory>

#include <ALIEN/AlienExternalPackagesPrecomp.h>

#include <alien/expression/solver/solver_stats/SolverStater.h>
#include <alien/core/backend/LinearSolverT.h>

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>

#include <ALIEN/Kernels/Hypre/HypreBackEnd.h>
#include <ALIEN/Kernels/Hypre/DataStructure/HypreVector.h>
#include <ALIEN/Kernels/Hypre/DataStructure/HypreMatrix.h>
#include <ALIEN/Kernels/Hypre/DataStructure/HypreInternal.h>
#include "HypreOptionTypes.h"

#include <ALIEN/axl/HypreSolver_IOptions.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Compile HypreLinearSolver.
template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearSolver<BackEnd::tag::hypre>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HypreInternalLinearSolver::HypreInternalLinearSolver(Arccore::MessagePassing::IMessagePassingMng* pm, IOptionsHypreSolver* options)
  : m_parallel_mng(pm)
  , m_options(options)
{
}

/*---------------------------------------------------------------------------*/

HypreInternalLinearSolver::~HypreInternalLinearSolver()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearSolver::init()
{}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearSolver::
updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm ALIEN_UNUSED_PARAM)
{
  ;
}

/*---------------------------------------------------------------------------*/

void HypreInternalLinearSolver::end()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearSolver::checkError(const Arccore::String & msg, int ierr, int skipError) const
{
  if (ierr != 0 and (ierr & ~skipError) != 0) {
    char hypre_error_msg[256];
    HYPRE_DescribeError(ierr,hypre_error_msg);
    alien_fatal([&] {
      cout() << msg << " failed : " << hypre_error_msg << "[code=" << ierr << "]";
    });
  }
}

/*---------------------------------------------------------------------------*/

bool
HypreInternalLinearSolver::solve(
    const HypreMatrix& A, const HypreVector& b, HypreVector& x)
{
  using namespace Alien;
  using namespace Alien::Internal;

  const HYPRE_IJMatrix& ij_matrix = A.internal()->internal();
  const HYPRE_IJVector& bij_vector = b.internal()->internal();
  HYPRE_IJVector& xij_vector = x.internal()->internal();

  // Macro "pratique" en attendant de trouver mieux
#ifdef VALUESTRING
#error Already defined macro VALUESTRING
#endif
#define VALUESTRING(option) ((option).enumValues()->nameOfValue((option)(),""))

  // Construction du système linéaire à résoudre au format
  // Le builder connait la strcuture interne garce au visiteur de l'init

  int output_level = m_options->verbose() ? 1 : 0;

  HYPRE_Solver solver = nullptr;
  HYPRE_Solver preconditioner = nullptr;

  // acces aux fonctions du preconditionneur
  HYPRE_PtrToParSolverFcn precond_solve_function = NULL;
  HYPRE_PtrToParSolverFcn precond_setup_function = NULL;
  int (*precond_destroy_function)(HYPRE_Solver) = NULL;

  auto* pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(A.getParallelMng());
  MPI_Comm comm = (*static_cast<const MPI_Comm*>(pm->getMPIComm()) == MPI_COMM_NULL)
      ? MPI_COMM_WORLD
      : *static_cast<const MPI_Comm*>(pm->getMPIComm());

  std::string precond_name = "undefined";
  switch (m_options->preconditioner()) {
  case HypreOptionTypes::NoPC:
    precond_name = "none";
      // precond_destroy_function = NULL;
      break;
    case HypreOptionTypes::DiagPC:
      precond_name = "diag";
      //checkError("Hypre diagonal preconditioner",HYPRE_BoomerAMGCreate(&preconditioner));
      precond_solve_function = HYPRE_ParCSRDiagScale;
      precond_setup_function = HYPRE_ParCSRDiagScaleSetup;
      break;
    case HypreOptionTypes::AMGPC:
      precond_name = "amg";
      checkError("Hypre AMG preconditioner",HYPRE_BoomerAMGCreate(&preconditioner));
      precond_solve_function = HYPRE_BoomerAMGSolve;
      precond_setup_function = HYPRE_BoomerAMGSetup;
      precond_destroy_function = HYPRE_BoomerAMGDestroy;
      break;
    case HypreOptionTypes::ParaSailsPC:
      precond_name = "parasails";
      checkError("Hypre ParaSails preconditioner",HYPRE_ParaSailsCreate(comm, &preconditioner));
      precond_solve_function = HYPRE_ParaSailsSolve;
      precond_setup_function = HYPRE_ParaSailsSetup;
      precond_destroy_function = HYPRE_ParaSailsDestroy;
      break;
    case HypreOptionTypes::EuclidPC:
      precond_name = "euclid";
      checkError("Hypre Euclid preconditioner",HYPRE_EuclidCreate(comm, &preconditioner));
      precond_solve_function = HYPRE_EuclidSolve;
      precond_setup_function = HYPRE_EuclidSetup;
      precond_destroy_function = HYPRE_EuclidDestroy;
      break;
    default:
      alien_fatal([&] {
        cout() << "Undefined Hypre preconditioner option";
      });
      break ;
    }

  // acces aux fonctions du solveur
  //int (*solver_set_logging_function)(HYPRE_Solver,int) = NULL;
  int (*solver_set_print_level_function)(HYPRE_Solver,int) = NULL;
  int (*solver_set_tol_function)(HYPRE_Solver,double) = NULL;
  int (*solver_set_precond_function)(HYPRE_Solver,HYPRE_PtrToParSolverFcn,HYPRE_PtrToParSolverFcn,HYPRE_Solver) = NULL;
  int (*solver_setup_function)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector) = NULL;
  int (*solver_solve_function)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector) = NULL;
  int (*solver_get_num_iterations_function)(HYPRE_Solver,int*) = NULL;
  int (*solver_get_final_relative_residual_function)(HYPRE_Solver,double*) = NULL;
  int (*solver_destroy_function)(HYPRE_Solver) = NULL;

  int max_it =  m_options->numIterationsMax();
  double rtol = m_options->stopCriteriaValue() ;

  std::string solver_name = "undefined";
  switch (m_options->solver())
  {
  case HypreOptionTypes::AMG:
	  solver_name = "amg";
	  checkError("Hypre AMG solver",HYPRE_BoomerAMGCreate(&solver));
	  if (output_level>0)
		  checkError("Hypre AMG SetDebugFlag",HYPRE_BoomerAMGSetDebugFlag(solver,1));
	  // Former configuration of Hypre (defaults change from 2.10 to 2.14)
	  /*
	  HYPRE_BoomerAMGSetCoarsenType(solver, 6);
	  HYPRE_BoomerAMGSetInterpType(solver, 0);
	  HYPRE_BoomerAMGSetRelaxType(solver, 3);
	  HYPRE_BoomerAMGSetRelaxOrder(solver, 1);
	  */
	  checkError("Hypre AMG solver SetMaxIter",HYPRE_BoomerAMGSetMaxIter(solver,max_it));
	  //solver_set_logging_function = HYPRE_BoomerAMGSetLogging;
	  solver_set_print_level_function = HYPRE_BoomerAMGSetPrintLevel;
	  solver_set_tol_function = HYPRE_BoomerAMGSetTol;
	  // solver_set_precond_function = NULL;
	  solver_setup_function = HYPRE_BoomerAMGSetup;
	  solver_solve_function = HYPRE_BoomerAMGSolve;
	  solver_get_num_iterations_function = HYPRE_BoomerAMGGetNumIterations;
	  solver_get_final_relative_residual_function = HYPRE_BoomerAMGGetFinalRelativeResidualNorm;
	  solver_destroy_function = HYPRE_BoomerAMGDestroy;
	  break;
    case HypreOptionTypes::GMRES:
    	solver_name = "gmres";
      checkError("Hypre GMRES solver",HYPRE_ParCSRGMRESCreate(comm, &solver));
      checkError("Hypre GMRES solver SetMaxIter",HYPRE_ParCSRGMRESSetMaxIter(solver,max_it));
      //solver_set_logging_function = HYPRE_ParCSRGMRESSetLogging;
      solver_set_print_level_function = HYPRE_ParCSRGMRESSetPrintLevel;
      solver_set_tol_function = HYPRE_ParCSRGMRESSetTol;
      solver_set_precond_function = HYPRE_ParCSRGMRESSetPrecond;
      solver_setup_function = HYPRE_ParCSRGMRESSetup;
      solver_solve_function = HYPRE_ParCSRGMRESSolve;
      solver_get_num_iterations_function = HYPRE_ParCSRGMRESGetNumIterations;
      solver_get_final_relative_residual_function = HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm;
      solver_destroy_function = HYPRE_ParCSRGMRESDestroy;
      break;
    case HypreOptionTypes::CG:
      solver_name = "cg";
      checkError("Hypre CG solver",HYPRE_ParCSRPCGCreate(comm, &solver));
      checkError("Hypre BiCGStab solver SetMaxIter",HYPRE_ParCSRPCGSetMaxIter(solver,max_it));
      //solver_set_logging_function = HYPRE_ParCSRPCGSetLogging;
      solver_set_print_level_function = HYPRE_ParCSRPCGSetPrintLevel;
      solver_set_tol_function = HYPRE_ParCSRPCGSetTol;
      solver_set_precond_function = HYPRE_ParCSRPCGSetPrecond;
      solver_setup_function = HYPRE_ParCSRPCGSetup;
      solver_solve_function = HYPRE_ParCSRPCGSolve;
      solver_get_num_iterations_function = HYPRE_ParCSRPCGGetNumIterations;
      solver_get_final_relative_residual_function = HYPRE_ParCSRPCGGetFinalRelativeResidualNorm;
      solver_destroy_function = HYPRE_ParCSRPCGDestroy;
      break;
    case HypreOptionTypes::BiCGStab:
      solver_name = "bicgs";
      checkError("Hypre BiCGStab solver",HYPRE_ParCSRBiCGSTABCreate(comm, &solver));
      checkError("Hypre BiCGStab solver SetMaxIter",HYPRE_ParCSRBiCGSTABSetMaxIter(solver,max_it));
      //solver_set_logging_function = HYPRE_ParCSRBiCGSTABSetLogging;
      solver_set_print_level_function = HYPRE_ParCSRBiCGSTABSetPrintLevel;
      solver_set_tol_function = HYPRE_ParCSRBiCGSTABSetTol;
      solver_set_precond_function = HYPRE_ParCSRBiCGSTABSetPrecond;
      solver_setup_function = HYPRE_ParCSRBiCGSTABSetup;
      solver_solve_function = HYPRE_ParCSRBiCGSTABSolve;
      solver_get_num_iterations_function = HYPRE_ParCSRBiCGSTABGetNumIterations;
      solver_get_final_relative_residual_function = HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm;
      solver_destroy_function = HYPRE_ParCSRBiCGSTABDestroy;
      break;
    case HypreOptionTypes::Hybrid:
      solver_name = "hybrid";
      checkError("Hypre Hybrid solver",HYPRE_ParCSRHybridCreate(&solver));
      // checkError("Hypre Hybrid solver SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,1)); // PCG
      checkError("Hypre Hybrid solver SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,2)); // GMRES
      // checkError("Hypre Hybrid solver SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,3)); // BiCGSTab
      checkError("Hypre Hybrid solver SetDiagMaxIter",HYPRE_ParCSRHybridSetDSCGMaxIter(solver,max_it));
      checkError("Hypre Hybrid solver SetPCMaxIter",HYPRE_ParCSRHybridSetPCGMaxIter(solver,max_it));
      //solver_set_logging_function = HYPRE_ParCSRHybridSetLogging;
      solver_set_print_level_function = HYPRE_ParCSRHybridSetPrintLevel;
      solver_set_tol_function = HYPRE_ParCSRHybridSetTol;
      solver_set_precond_function = NULL; // HYPRE_ParCSRHybridSetPrecond; // SegFault si utilise un prï¿œconditionneur !
      solver_setup_function = HYPRE_ParCSRHybridSetup;
      solver_solve_function = HYPRE_ParCSRHybridSolve;
      solver_get_num_iterations_function = HYPRE_ParCSRHybridGetNumIterations;
      solver_get_final_relative_residual_function = HYPRE_ParCSRHybridGetFinalRelativeResidualNorm;
      solver_destroy_function = HYPRE_ParCSRHybridDestroy;
      break;
    default:
      alien_fatal([&] {
        cout() << "Undefined solver option";
      });
      break ;
    }

  if (solver_set_precond_function){
    if (precond_solve_function){
      checkError("Hypre "+solver_name+" solver SetPreconditioner",
          (*solver_set_precond_function)(solver,precond_solve_function,
          precond_setup_function,preconditioner));
    }
  }
  else
    {
      if (precond_solve_function)
        {
          alien_fatal([&] {
            cout() << "Hypre " << solver_name << " solver cannot accept preconditioner";
          });
        }
    }

  checkError("Hypre "+solver_name+" solver SetStopCriteria",(*solver_set_tol_function)(solver,rtol));

  if (output_level>0)
    {
      checkError("Hypre "+solver_name+" solver Setlogging",(*solver_set_print_level_function)(solver,1));
      checkError("Hypre "+solver_name+" solver SetPrintLevel",(*solver_set_print_level_function)(solver,3));
    }

  HYPRE_ParCSRMatrix par_a;
  HYPRE_ParVector par_rhs, par_x;
  checkError("Hypre Matrix GetObject",HYPRE_IJMatrixGetObject(ij_matrix, (void **) &par_a));
  checkError("Hypre RHS Vector GetObject",HYPRE_IJVectorGetObject(bij_vector, (void **) &par_rhs));
  checkError("Hypre Unknown Vector GetObject",HYPRE_IJVectorGetObject(xij_vector, (void **) &par_x));

  checkError("Hypre "+solver_name+" solver Setup",(*solver_setup_function)(solver, par_a, par_rhs, par_x));
  m_status.succeeded = ((*solver_solve_function)(solver, par_a, par_rhs, par_x) == 0);

  checkError("Hypre "+solver_name+" solver GetNumIterations",(*solver_get_num_iterations_function)(solver,&m_status.iteration_count));
  checkError("Hypre "+solver_name+" solver GetFinalResidual",(*solver_get_final_relative_residual_function)(solver,&m_status.residual));

  checkError("Hypre "+solver_name+" solver Destroy",(*solver_destroy_function)(solver));
  if (precond_destroy_function)
    checkError("Hypre "+precond_name+" preconditioner Destroy",(*precond_destroy_function)(preconditioner));

  return m_status.succeeded;

#undef VALUESTRING
}

/*---------------------------------------------------------------------------*/

const Alien::SolverStatus&
HypreInternalLinearSolver::getStatus() const
{
  return m_status;
}

/*---------------------------------------------------------------------------*/

std::shared_ptr<ILinearAlgebra>
HypreInternalLinearSolver::algebra() const
{
  return std::shared_ptr<ILinearAlgebra>();
  //return std::shared_ptr<ILinearAlgebra>(new HypreLinearAlgebra());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IInternalLinearSolver<HypreMatrix, HypreVector>*
HypreInternalLinearSolverFactory(Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsHypreSolver* options)
{
  return new HypreInternalLinearSolver(p_mng, options);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
