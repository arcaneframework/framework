#include "HypreInternalLinearSolver.h"

#include <memory>

#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/LinearSolverT.h>

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>

#include <alien/core/backend/SolverFabricRegisterer.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>
#include "HypreOptionTypes.h"

#include <ALIEN/axl/HypreSolver_IOptions.h>

#include <alien/kernels/hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/axl/HypreSolver_IOptions.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>

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
bool HypreInternalLinearSolver::m_library_plugin_is_initialized = false ;

std::unique_ptr<HypreLibrary> HypreInternalLinearSolver::m_library_plugin ;

HypreLibrary::HypreLibrary()
{
#if HYPRE_HAVE_HYPRE_INIT
  HYPRE_Init() ;
#endif
}

HypreLibrary::~HypreLibrary()
{
#if HYPRE_HAVE_HYPRE_FINALIZE
  HYPRE_Finalize() ;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HypreInternalLinearSolver::HypreInternalLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* pm, IOptionsHypreSolver* options)
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
{
  if(HypreInternalLinearSolver::m_library_plugin_is_initialized) return ;
#ifdef HYPRE_USING_CUDA
  if(m_options->useGpu() )
  {
    hypre_SetDevice(m_gpu_device_id,nullptr);
    HypreInternalLinearSolver::m_library_plugin.reset(new HypreLibrary()) ;
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    /* setup AMG on GPUs */
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
    /* use hypre's SpGEMM instead of cuSPARSE */
    HYPRE_SetSpGemmUseCusparse(false);
    /* use GPU RNG */
    HYPRE_SetUseGpuRand(true);
    bool useHypreGpuMemPool = false ;
    bool useUmpireGpuMemPool = false ;
    if (useHypreGpuMemPool)
    {
      /* use hypre's GPU memory pool */
      //HYPRE_SetGPUMemoryPoolSize(bin_growth, min_bin, max_bin, max_bytes);
    }
    else if (useUmpireGpuMemPool)
     {
       /* or use Umpire GPU memory pool */
       //HYPRE_SetUmpireUMPoolName("HYPRE_UM_POOL_TEST");
       //HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE_POOL_TEST");
     }
   }
   else
#endif
   {
     HypreInternalLinearSolver::m_library_plugin.reset(new HypreLibrary()) ;
   }
   HypreInternalLinearSolver::m_library_plugin_is_initialized = true ;
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearSolver::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm ALIEN_UNUSED_PARAM)
{
  ;
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearSolver::end()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
HypreInternalLinearSolver::checkError(
    const Arccore::String& msg, int ierr, int skipError) const
{
  if (ierr != 0 and (ierr & ~skipError) != 0) {
    char hypre_error_msg[256];
    HYPRE_DescribeError(ierr, hypre_error_msg);
    alien_info([&] {
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
#define VALUESTRING(option) ((option).enumValues()->nameOfValue((option)(), ""))

  // Construction du système linéaire à résoudre au format
  // Le builder connait la strcuture interne garce au visiteur de l'init

  int output_level = m_options->verbose() ? 1 : 0;

  HYPRE_Solver solver = nullptr;
  HYPRE_Solver preconditioner = nullptr;

  // acces aux fonctions du preconditionneur
  HYPRE_PtrToParSolverFcn precond_solve_function = NULL;
  HYPRE_PtrToParSolverFcn precond_setup_function = NULL;
  int (*precond_destroy_function)(HYPRE_Solver) = NULL;

  auto* pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(
      A.getParallelMng());
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
    // checkError("Hypre diagonal preconditioner",HYPRE_BoomerAMGCreate(&preconditioner));
    precond_solve_function = HYPRE_ParCSRDiagScale;
    precond_setup_function = HYPRE_ParCSRDiagScaleSetup;
    break;
  case HypreOptionTypes::AMGPC:
    precond_name = "amg";
    checkError("Hypre AMG preconditioner", HYPRE_BoomerAMGCreate(&preconditioner));
    precond_solve_function = HYPRE_BoomerAMGSolve;
    precond_setup_function = HYPRE_BoomerAMGSetup;
    precond_destroy_function = HYPRE_BoomerAMGDestroy;
    {
      int coarsening_opt = 8;
      int interpolation_type = 7;
      double StrongThreshold = 0.15;
      int amg_debug_flag = 0;
      int bicgs_debug_flag =0;
      int ierr = 0;
      ierr = HYPRE_BoomerAMGSetMaxIter(preconditioner,2) ;//Sophie::
      if( ierr == HYPRE_ERROR_CONV){
          ierr = 0;
      }else if(ierr == HYPRE_ERROR_GENERIC){
          printf("HYPRE_ERROR_GENERIC while calling HYPRE_BoomerAMGSetMaxIter with default value\n");
      }else if (ierr == HYPRE_ERROR_MEMORY){
          printf("HYPRE_ERROR_MEMORY while calling HYPRE_BoomerAMGSetMaxIter with default value\n");
      }else if(ierr == HYPRE_ERROR_ARG){
          printf("HYPRE_ERROR_ARG while calling HYPRE_BoomerAMGSetMaxIter with default value\n");
      }
      if(ierr) {
          printf("Error while calling HYPRE_BoomerAMGSetMaxIter with default value\n");
          exit(0);
      }
      ierr = HYPRE_BoomerAMGSetTol(preconditioner,1.e-7) ;

      ierr = HYPRE_BoomerAMGSetMaxLevels(preconditioner,25) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetMaxLevels with default value\n"); exit(0);}
      ierr = HYPRE_BoomerAMGSetMaxRowSum(preconditioner,0.9) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetMaxRowSum with default value\n"); exit(0);}
      ierr = HYPRE_BoomerAMGSetCycleType(preconditioner,1) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetCycleType with default value\n"); exit(0);}
      ierr = HYPRE_BoomerAMGSetRelaxType(preconditioner,3) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetRelaxType\n"); exit(0);}

       ierr = HYPRE_BoomerAMGSetCoarsenType(preconditioner,coarsening_opt) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetCoarsenType\n"); exit(0);}

       ierr = HYPRE_BoomerAMGSetNumSweeps(preconditioner,1) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetNumSweeps\n"); exit(0);}

       ierr = HYPRE_BoomerAMGSetInterpType(preconditioner,interpolation_type) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetInterpType\n"); exit(0);}

       ierr = HYPRE_BoomerAMGSetSmoothNumLevels(preconditioner,0) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetSmoothNumLevels\n"); exit(0);}

       ierr = HYPRE_BoomerAMGSetSmoothType(preconditioner,-1) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetSmoothType\n"); exit(0);}


       ierr = HYPRE_BoomerAMGSetStrongThreshold(preconditioner,StrongThreshold) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetStrongThreshold\n"); exit(0);}

       ierr = HYPRE_BoomerAMGSetMeasureType(preconditioner,0) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetMeasureType with default value\n"); exit(0);}
       ierr = HYPRE_BoomerAMGSetAggNumLevels(preconditioner,0) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetAggNumLevels with default value\n"); exit(0);}
       ierr = HYPRE_BoomerAMGSetNumPaths(preconditioner,1) ; if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) { printf("Error while calling HYPRE_BoomerAMGSetNumPaths with default value\n"); exit(0);}
       if (output_level > 2) {
          checkError("Hypre " + precond_name + " solver Setlogging",
                      HYPRE_BoomerAMGSetLogging(preconditioner,1));
          checkError("Hypre " + precond_name + " solver SetPrintLevel",
                      HYPRE_BoomerAMGSetPrintLevel(preconditioner, 3));
       }
    }
    break;
  case HypreOptionTypes::ParaSailsPC:
    precond_name = "parasails";
    checkError(
        "Hypre ParaSails preconditioner", HYPRE_ParaSailsCreate(comm, &preconditioner));
    precond_solve_function = HYPRE_ParaSailsSolve;
    precond_setup_function = HYPRE_ParaSailsSetup;
    precond_destroy_function = HYPRE_ParaSailsDestroy;
    break;
  case HypreOptionTypes::EuclidPC:
    precond_name = "euclid";
    checkError("Hypre Euclid preconditioner", HYPRE_EuclidCreate(comm, &preconditioner));
    precond_solve_function = HYPRE_EuclidSolve;
    precond_setup_function = HYPRE_EuclidSetup;
    precond_destroy_function = HYPRE_EuclidDestroy;
    break;
  default:
    alien_fatal([&] { cout() << "Undefined Hypre preconditioner option"; });
    break;
  }

  // acces aux fonctions du solveur
  // int (*solver_set_logging_function)(HYPRE_Solver,int) = NULL;
  int (*solver_set_print_level_function)(HYPRE_Solver, int) = NULL;
  int (*solver_set_tol_function)(HYPRE_Solver, double) = NULL;
  int (*solver_set_precond_function)(HYPRE_Solver, HYPRE_PtrToParSolverFcn,
      HYPRE_PtrToParSolverFcn, HYPRE_Solver) = NULL;
  int (*solver_setup_function)(
      HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector) = NULL;
  int (*solver_solve_function)(
      HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector) = NULL;
  int (*solver_get_num_iterations_function)(HYPRE_Solver, int*) = NULL;
  int (*solver_get_final_relative_residual_function)(HYPRE_Solver, double*) = NULL;
  int (*solver_destroy_function)(HYPRE_Solver) = NULL;

  int max_it = m_options->numIterationsMax();
  double rtol = m_options->stopCriteriaValue();

  std::string solver_name = "undefined";
  switch (m_options->solver()) {
  case HypreOptionTypes::AMG:
    solver_name = "amg";
    checkError("Hypre AMG solver", HYPRE_BoomerAMGCreate(&solver));
    if (output_level > 0)
      checkError("Hypre AMG SetDebugFlag", HYPRE_BoomerAMGSetDebugFlag(solver, 1));
    // Former configuration of Hypre (defaults change from 2.10 to 2.14)
    /*
    HYPRE_BoomerAMGSetCoarsenType(solver, 6);
    HYPRE_BoomerAMGSetInterpType(solver, 0);
    HYPRE_BoomerAMGSetRelaxType(solver, 3);
    HYPRE_BoomerAMGSetRelaxOrder(solver, 1);
    */
    checkError("Hypre AMG solver SetMaxIter", HYPRE_BoomerAMGSetMaxIter(solver, max_it));
    // solver_set_logging_function = HYPRE_BoomerAMGSetLogging;
    solver_set_print_level_function = HYPRE_BoomerAMGSetPrintLevel;
    solver_set_tol_function = HYPRE_BoomerAMGSetTol;
    // solver_set_precond_function = NULL;
    solver_setup_function = HYPRE_BoomerAMGSetup;
    solver_solve_function = HYPRE_BoomerAMGSolve;
    solver_get_num_iterations_function = HYPRE_BoomerAMGGetNumIterations;
    solver_get_final_relative_residual_function =
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm;
    solver_destroy_function = HYPRE_BoomerAMGDestroy;
    break;
  case HypreOptionTypes::GMRES:
    solver_name = "gmres";
    checkError("Hypre GMRES solver", HYPRE_ParCSRGMRESCreate(comm, &solver));
    checkError(
        "Hypre GMRES solver SetMaxIter", HYPRE_ParCSRGMRESSetMaxIter(solver, max_it));
    // solver_set_logging_function = HYPRE_ParCSRGMRESSetLogging;
    solver_set_print_level_function = HYPRE_ParCSRGMRESSetPrintLevel;
    solver_set_tol_function = HYPRE_ParCSRGMRESSetTol;
    solver_set_precond_function = HYPRE_ParCSRGMRESSetPrecond;
    solver_setup_function = HYPRE_ParCSRGMRESSetup;
    solver_solve_function = HYPRE_ParCSRGMRESSolve;
    solver_get_num_iterations_function = HYPRE_ParCSRGMRESGetNumIterations;
    solver_get_final_relative_residual_function =
        HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm;
    solver_destroy_function = HYPRE_ParCSRGMRESDestroy;
    break;
  case HypreOptionTypes::CG:
    solver_name = "cg";
    checkError("Hypre CG solver", HYPRE_ParCSRPCGCreate(comm, &solver));
    checkError(
        "Hypre BiCGStab solver SetMaxIter", HYPRE_ParCSRPCGSetMaxIter(solver, max_it));
    // solver_set_logging_function = HYPRE_ParCSRPCGSetLogging;
    solver_set_print_level_function = HYPRE_ParCSRPCGSetPrintLevel;
    solver_set_tol_function = HYPRE_ParCSRPCGSetTol;
    solver_set_precond_function = HYPRE_ParCSRPCGSetPrecond;
    solver_setup_function = HYPRE_ParCSRPCGSetup;
    solver_solve_function = HYPRE_ParCSRPCGSolve;
    solver_get_num_iterations_function = HYPRE_ParCSRPCGGetNumIterations;
    solver_get_final_relative_residual_function =
        HYPRE_ParCSRPCGGetFinalRelativeResidualNorm;
    solver_destroy_function = HYPRE_ParCSRPCGDestroy;
    break;
  case HypreOptionTypes::BiCGStab:
    solver_name = "bicgs";
    checkError("Hypre BiCGStab solver", HYPRE_ParCSRBiCGSTABCreate(comm, &solver));
    checkError("Hypre BiCGStab solver SetMaxIter",
        HYPRE_ParCSRBiCGSTABSetMaxIter(solver, max_it));
    // solver_set_logging_function = HYPRE_ParCSRBiCGSTABSetLogging;
    solver_set_print_level_function = HYPRE_ParCSRBiCGSTABSetPrintLevel;
    solver_set_tol_function = HYPRE_ParCSRBiCGSTABSetTol;
    solver_set_precond_function = HYPRE_ParCSRBiCGSTABSetPrecond;
    solver_setup_function = HYPRE_ParCSRBiCGSTABSetup;
    solver_solve_function = HYPRE_ParCSRBiCGSTABSolve;
    solver_get_num_iterations_function = HYPRE_ParCSRBiCGSTABGetNumIterations;
    solver_get_final_relative_residual_function =
        HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm;
    solver_destroy_function = HYPRE_ParCSRBiCGSTABDestroy;
    break;
  case HypreOptionTypes::Hybrid:
    solver_name = "hybrid";
    checkError("Hypre Hybrid solver", HYPRE_ParCSRHybridCreate(&solver));
    // checkError("Hypre Hybrid solver
    // SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,1)); // PCG
    checkError("Hypre Hybrid solver SetSolverType",
        HYPRE_ParCSRHybridSetSolverType(solver, 2)); // GMRES
    // checkError("Hypre Hybrid solver
    // SetSolverType",HYPRE_ParCSRHybridSetSolverType(solver,3)); // BiCGSTab
    checkError("Hypre Hybrid solver SetDiagMaxIter",
        HYPRE_ParCSRHybridSetDSCGMaxIter(solver, max_it));
    checkError("Hypre Hybrid solver SetPCMaxIter",
        HYPRE_ParCSRHybridSetPCGMaxIter(solver, max_it));
    // solver_set_logging_function = HYPRE_ParCSRHybridSetLogging;
    solver_set_print_level_function = HYPRE_ParCSRHybridSetPrintLevel;
    solver_set_tol_function = HYPRE_ParCSRHybridSetTol;
    solver_set_precond_function = NULL; // HYPRE_ParCSRHybridSetPrecond; // SegFault si
                                        // utilise un prï¿œconditionneur !
    solver_setup_function = HYPRE_ParCSRHybridSetup;
    solver_solve_function = HYPRE_ParCSRHybridSolve;
    solver_get_num_iterations_function = HYPRE_ParCSRHybridGetNumIterations;
    solver_get_final_relative_residual_function =
        HYPRE_ParCSRHybridGetFinalRelativeResidualNorm;
    solver_destroy_function = HYPRE_ParCSRHybridDestroy;
    break;
  default:
    alien_fatal([&] { cout() << "Undefined solver option"; });
    break;
  }

  if (solver_set_precond_function) {
    if (precond_solve_function) {
      checkError("Hypre " + solver_name + " solver SetPreconditioner",
          (*solver_set_precond_function)(
              solver, precond_solve_function, precond_setup_function, preconditioner));
    }
  } else {
    if (precond_solve_function) {
      alien_fatal([&] {
        cout() << "Hypre " << solver_name << " solver cannot accept preconditioner";
      });
    }
  }

  checkError("Hypre " + solver_name + " solver SetStopCriteria",
      (*solver_set_tol_function)(solver, rtol));

  if (output_level > 0) {
    checkError("Hypre " + solver_name + " solver Setlogging",
        (*solver_set_print_level_function)(solver, 1));
    checkError("Hypre " + solver_name + " solver SetPrintLevel",
        (*solver_set_print_level_function)(solver, 3));
  }

  HYPRE_ParCSRMatrix par_a;
  HYPRE_ParVector par_rhs, par_x;
  checkError(
      "Hypre Matrix GetObject", HYPRE_IJMatrixGetObject(ij_matrix, (void**)&par_a));
  checkError("Hypre RHS Vector GetObject",
      HYPRE_IJVectorGetObject(bij_vector, (void**)&par_rhs));
  checkError("Hypre Unknown Vector GetObject",
      HYPRE_IJVectorGetObject(xij_vector, (void**)&par_x));

  checkError("Hypre " + solver_name + " solver Setup",
      (*solver_setup_function)(solver, par_a, par_rhs, par_x));
  m_status.succeeded = ((*solver_solve_function)(solver, par_a, par_rhs, par_x) == 0);

  checkError("Hypre " + solver_name + " solver GetNumIterations",
      (*solver_get_num_iterations_function)(solver, &m_status.iteration_count));
  checkError("Hypre " + solver_name + " solver GetFinalResidual",
      (*solver_get_final_relative_residual_function)(solver, &m_status.residual));

  checkError(
      "Hypre " + solver_name + " solver Destroy", (*solver_destroy_function)(solver));
  if (precond_destroy_function)
    checkError("Hypre " + precond_name + " preconditioner Destroy",
        (*precond_destroy_function)(preconditioner));

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
  // return std::shared_ptr<ILinearAlgebra>(new HypreLinearAlgebra());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IInternalLinearSolver<HypreMatrix, HypreVector>*
HypreInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsHypreSolver* options)
{
  return new HypreInternalLinearSolver(p_mng, options);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class SolverFabric<Alien::BackEnd::tag::hypre>
: public ISolverFabric
{
public :

  BackEndId backend() const {
     return "hypre" ;
  }

  void
  add_options(CmdLineOptionDescType& cmdline_options) const
  {
    using namespace boost::program_options;
    options_description desc("HYPRE options");
    desc.add_options()("hypre-solver", value<std::string>()->default_value("bicgs"),"solver algo name : amg cg gmres bicgstab")
                      ("hypre-precond", value<std::string>()->default_value("none"),"preconditioner none diag amg parasails euclid");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  Alien::ILinearSolver* _create(OptionT const& options,Alien::IMessagePassingMng* pm) const
  {
    double tol = get<double>(options,"tol");
    int max_iter = get<int>(options,"max-iter");

    std::string solver_type_s =  get<std::string>(options,"hypre-solver");
    HypreOptionTypes::eSolver solver_type =
        OptionsHypreSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s =  get<std::string>(options,"hypre-precond");
    HypreOptionTypes::ePreconditioner precond_type =
        OptionsHypreSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    using namespace HypreSolverOptionsNames;
    auto solver_options = std::make_shared<StrongOptionsHypreSolver>(
        _numIterationsMax = max_iter, _stopCriteriaValue = tol, _solver = solver_type,
        _preconditioner = precond_type);
    // service
   return new Alien::HypreLinearSolver(pm, solver_options);
  }

  Alien::ILinearSolver* create(CmdLineOptionType const& options,Alien::IMessagePassingMng* pm) const
  {
    return _create(options,pm) ;
  }

  Alien::ILinearSolver* create(JsonOptionType const& options,Alien::IMessagePassingMng* pm) const
  {
    return _create(options,pm) ;
  }

};

typedef SolverFabric<Alien::BackEnd::tag::hypre> HypreSolverFabric ;
REGISTER_SOLVER_FABRIC(HypreSolverFabric);
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
