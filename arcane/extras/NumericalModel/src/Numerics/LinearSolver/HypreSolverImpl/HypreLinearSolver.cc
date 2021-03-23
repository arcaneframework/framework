// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Utils/Utils.h"

#include <arcane/ArcaneVersion.h>

#include <HYPRE_utilities.h>
#include "_hypre_utilities.h" // pour HYPRE_ERROR_CONV
#include <HYPRE.h>
#include <HYPRE_parcsr_mv.h>

#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include "Numerics/LinearSolver/ILinearSystemVisitor.h"
#include "Numerics/LinearSolver/ILinearSystemBuilder.h"
#include "Numerics/LinearSolver/ILinearSystem.h"
#include "Numerics/LinearSolver/ILinearSolver.h"
#include "Numerics/LinearSolver/HypreSolverImpl/HypreLinearSystem.h"
#include "Numerics/LinearSolver/HypreSolverImpl/HypreLinearSolver.h"
#include "Numerics/LinearSolver/HypreSolverImpl/HypreInternal.h"

#ifdef _MPI
#define MPICH_SKIP_MPICXX 1
#include "mpi.h"
#endif

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HypreLinearSolver::
HypreLinearSolver(const ServiceBuildInfo & sbi) 
 : ArcaneHypreSolverObject(sbi) 
 , m_system_is_built(false)
 , m_system_is_locked(false)
{
  m_system = NULL;
  m_builder = NULL;
}

/*---------------------------------------------------------------------------*/

HypreLinearSolver::
~HypreLinearSolver()
{
  freeLinearSystem();
}

/*---------------------------------------------------------------------------*/

void HypreLinearSolver::setLinearSystemBuilder(ILinearSystemBuilder * builder)
{
  m_builder = builder;
}


/*---------------------------------------------------------------------------*/

void HypreLinearSolver::init()
{
  freeLinearSystem();
  updateLinearSystem();

  if (m_builder == NULL)
    fatal() << "LinearSystemBuilder must be set before initializing LinearSolver";

  m_builder->connect(m_system);
  m_builder->init();
}

/*---------------------------------------------------------------------------*/

void HypreLinearSolver::start()
{
  updateLinearSystem() ;
  m_system->start();
}

/*---------------------------------------------------------------------------*/

void 
HypreLinearSolver::
updateLinearSystem()
{
  if (m_system==NULL)
  {
    m_system = new HypreLinearSystem(this);
    m_system->init();
  }
}

/*---------------------------------------------------------------------------*/

void 
HypreLinearSolver::
freeLinearSystem()
{
  delete m_system;
  m_system = NULL;
  m_system_is_built = false ;
  m_system_is_locked = false ;
}

/*---------------------------------------------------------------------------*/

ILinearSystem * 
HypreLinearSolver::
getLinearSystem()
{
  updateLinearSystem();

  return m_system ;
}

/*---------------------------------------------------------------------------*/

bool 
HypreLinearSolver::
buildLinearSystem()
{
  bool flag = m_system->accept(m_builder) ;
  m_system_is_built = true ;
  m_system_is_locked = false ;
  return flag ;
}

/*---------------------------------------------------------------------------*/

bool 
HypreLinearSolver::
getSolution()
{
  return m_builder->commitSolution(m_system) ;
}

/*---------------------------------------------------------------------------*/

bool HypreLinearSolver::solve()
{
  // Macro "pratique" en attendant de trouver mieux
#ifdef VALUESTRING
#error Already defined macro VALUESTRING
#endif
#define VALUESTRING(option) ((option).enumValues()->nameOfValue((option)(),""))

  // Construction du système linéaire à résoudre au format
  // Le builder connait la strcuture interne garce au visiteur de l'init
  if(!m_system_is_built)
  {
    fatal()<<"Linear system is not built, buildLinearSystem() should be call first" ;
    m_system->accept(m_builder) ;
  }
   if(m_system_is_locked)
     fatal()<<"Error : try to solve a system already solved once without having been modified since" ;


  HYPRE_Solver solver;
  HYPRE_Solver preconditioner;

  // acces aux fonctions du preconditionneur
  HYPRE_PtrToParSolverFcn precond_solve_function = NULL;
  HYPRE_PtrToParSolverFcn precond_setup_function = NULL;
  int (*precond_destroy_function)(HYPRE_Solver) = NULL;

  switch (options()->preconditioner())
    {
    case HypreOptionTypes::NoPC:
      // precond_destroy_function = NULL;
      break;
    case HypreOptionTypes::AMGPC:
      checkError("Hypre AMG preconditioner",HYPRE_BoomerAMGCreate(&preconditioner));
      precond_solve_function = HYPRE_BoomerAMGSolve;
      precond_setup_function = HYPRE_BoomerAMGSetup;
      precond_destroy_function = HYPRE_BoomerAMGDestroy;
      break;
    case HypreOptionTypes::ParaSailsPC:
      checkError("Hypre ParaSails preconditioner",HYPRE_ParaSailsCreate(MPI_COMM_WORLD,&preconditioner));
      precond_solve_function = HYPRE_ParaSailsSolve;
      precond_setup_function = HYPRE_ParaSailsSetup;
      precond_destroy_function = HYPRE_ParaSailsDestroy;
      break;
    case HypreOptionTypes::EuclidPC:
      checkError("Hypre Euclid preconditioner",HYPRE_EuclidCreate(MPI_COMM_WORLD,&preconditioner));
      precond_solve_function = HYPRE_EuclidSolve;
      precond_setup_function = HYPRE_EuclidSetup;
      precond_destroy_function = HYPRE_EuclidDestroy;
      break;
    default:
      fatal() << "Undefined Hypre preconditioner option";
    }

  // acces aux fonctions du solveur
  int (*solver_set_logging_function)(HYPRE_Solver,int) = NULL;
  int (*solver_set_print_level_function)(HYPRE_Solver,int) = NULL;
  int (*solver_set_tol_function)(HYPRE_Solver,double) = NULL;
  int (*solver_set_precond_function)(HYPRE_Solver,HYPRE_PtrToParSolverFcn,HYPRE_PtrToParSolverFcn,HYPRE_Solver) = NULL;
  int (*solver_setup_function)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector) = NULL;
  int (*solver_solve_function)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector) = NULL;
  int (*solver_get_num_iterations_function)(HYPRE_Solver,int*) = NULL;
  int (*solver_get_final_relative_residual_function)(HYPRE_Solver,double*) = NULL;
  int (*solver_destroy_function)(HYPRE_Solver) = NULL;
  
  switch (options()->solver())
    {
    case HypreOptionTypes::AMG:
      checkError("Hypre AMG solver",HYPRE_BoomerAMGCreate(&solver));
      if (options()->verbose())
        checkError("Hypre AMG SetDebugFlag",HYPRE_BoomerAMGSetDebugFlag(solver,1));
      checkError("Hypre AMG solver SetMaxIter",HYPRE_BoomerAMGSetMaxIter(solver,options()->numIterationsMax()));
      solver_set_logging_function = HYPRE_BoomerAMGSetLogging;
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
      checkError("Hypre GMRES solver",HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD,&solver));
      checkError("Hypre GMRES solver SetMaxIter",HYPRE_ParCSRGMRESSetMaxIter(solver,options()->numIterationsMax()));
      solver_set_logging_function = HYPRE_ParCSRGMRESSetLogging;
      solver_set_print_level_function = HYPRE_ParCSRGMRESSetPrintLevel;
      solver_set_tol_function = HYPRE_ParCSRGMRESSetTol;
      solver_set_precond_function = HYPRE_ParCSRGMRESSetPrecond;
      solver_setup_function = HYPRE_ParCSRGMRESSetup;
      solver_solve_function = HYPRE_ParCSRGMRESSolve;
      solver_get_num_iterations_function = HYPRE_ParCSRGMRESGetNumIterations;
      solver_get_final_relative_residual_function = HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm;
      solver_destroy_function = HYPRE_ParCSRGMRESDestroy;
      break;
    case HypreOptionTypes::BiCGStab:
      checkError("Hypre BiCGStab solver",HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD,&solver));
      checkError("Hypre BiCGStab solver SetMaxIter",HYPRE_ParCSRBiCGSTABSetMaxIter(solver,options()->numIterationsMax()));
      solver_set_logging_function = HYPRE_ParCSRBiCGSTABSetLogging;
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
      checkError("Hypre Hybrid solver",HYPRE_ParCSRHybridCreate(&solver));
      // checkError("Hypre Hybrid solver SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,1)); // PCG
      checkError("Hypre Hybrid solver SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,2)); // GMRES
      // checkError("Hypre Hybrid solver SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,3)); // BiCGSTab
      checkError("Hypre Hybrid solver SetDiagMaxIter",HYPRE_ParCSRHybridSetDSCGMaxIter(solver,options()->numIterationsMax()));
      checkError("Hypre Hybrid solver SetPCMaxIter",HYPRE_ParCSRHybridSetPCGMaxIter(solver,options()->numIterationsMax()));
      solver_set_logging_function = HYPRE_ParCSRHybridSetLogging;
      solver_set_print_level_function = HYPRE_ParCSRHybridSetPrintLevel;
      solver_set_tol_function = HYPRE_ParCSRHybridSetTol;
      solver_set_precond_function = NULL; // HYPRE_ParCSRHybridSetPrecond; // SegFault si utilise un préconditionneur !
      solver_setup_function = HYPRE_ParCSRHybridSetup;
      solver_solve_function = HYPRE_ParCSRHybridSolve;
      solver_get_num_iterations_function = HYPRE_ParCSRHybridGetNumIterations;
      solver_get_final_relative_residual_function = HYPRE_ParCSRHybridGetFinalRelativeResidualNorm;
      solver_destroy_function = HYPRE_ParCSRHybridDestroy;
      break;
    default:
      fatal() << "Undefined solver option";
    }

  if (solver_set_precond_function)
    {
      if (precond_solve_function) 
        {
          checkError("Hypre "+VALUESTRING(options()->solver)+" solver SetPreconditioner",(*solver_set_precond_function)(solver,precond_solve_function,precond_setup_function,preconditioner));
        }
    }
  else
    {
      if (precond_solve_function) 
        {
          fatal() << "Hypre " << VALUESTRING(options()->solver) << " solver cannot accept preconditioner";
        }
    }
  
  checkError("Hypre "+VALUESTRING(options()->solver)+" solver SetStopCriteria",(*solver_set_tol_function)(solver,options()->stopCriteriaValue()));

  if (options()->verbose())
    {
      checkError("Hypre "+VALUESTRING(options()->solver)+" solver Setlogging",(*solver_set_print_level_function)(solver,1));
      checkError("Hypre "+VALUESTRING(options()->solver)+" solver SetPrintLevel",(*solver_set_print_level_function)(solver,3));
    }

  HYPRE_ParCSRMatrix A;
  HYPRE_ParVector b, x;
  checkError("Hypre Matrix GetObject",HYPRE_IJMatrixGetObject(m_system->m_internal->m_ij_matrix, (void **) &A));
  checkError("Hypre RHS Vector GetObject",HYPRE_IJVectorGetObject(m_system->m_internal->m_bij_vector, (void **) &b));
  checkError("Hypre Unknown Vector GetObject",HYPRE_IJVectorGetObject(m_system->m_internal->m_xij_vector, (void **) &x));

  checkError("Hypre "+VALUESTRING(options()->solver)+" solver Setup",(*solver_setup_function)(solver, A, b, x));
  m_status.succeeded = ((*solver_solve_function)(solver, A, b, x) == 0);

  checkError("Hypre "+VALUESTRING(options()->solver)+" solver GetNumIterations",(*solver_get_num_iterations_function)(solver,&m_status.iteration_count));
  checkError("Hypre "+VALUESTRING(options()->solver)+" solver GetFinalResidual",(*solver_get_final_relative_residual_function)(solver,&m_status.residual));

  checkError("Hypre "+VALUESTRING(options()->solver)+" solver Destroy",(*solver_destroy_function)(solver));
  if (precond_destroy_function)
    checkError("Hypre "+VALUESTRING(options()->preconditioner)+" preconditioner Destroy",(*precond_destroy_function)(preconditioner));

  //lock linear system to prevent from solving it twice without changing it
  m_system_is_locked = true ;
  return m_status.succeeded;

#undef VALUESTRING
}

/*---------------------------------------------------------------------------*/

void HypreLinearSolver::end()
{
  //m_builder->end();
  m_system->end() ;
  freeLinearSystem();
  m_system_is_built = false ;
}

/*---------------------------------------------------------------------------*/

void HypreLinearSolver::checkError(const String & msg, int ierr, int skipError) const
{
  if (ierr != 0 and (ierr & ~skipError) != 0)
    {
      char hypre_error_msg[256];
      HYPRE_DescribeError(ierr,hypre_error_msg);
      fatal() << msg << " failed : " << hypre_error_msg << "[code=" << ierr << "]";
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HYPRESOLVER(HypreSolver,HypreLinearSolver);
