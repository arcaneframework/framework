// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

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

HypreLibrary::HypreLibrary(bool exec_on_device, bool use_device_memory, int device_id)
{
  if(exec_on_device)
  {
    m_device_id = device_id ;
    //hypre_SetDevice(m_device_id, nullptr);
  }
#if HYPRE_RELEASE_NUMBER >= 22900
  if (!HYPRE_Initialized()){
    HYPRE_Initialize();
  }
#elif HYPRE_RELEASE_NUMBER >= 22700
  HYPRE_Init();
#endif
#if HYPRE_RELEASE_NUMBER >= 22700
  if(exec_on_device)
  {
    m_exec_space = BackEnd::Exec::Device ;
    if(use_device_memory)
    {
      m_memory_type = BackEnd::Memory::Device ;
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    }
    else
    {
      m_memory_type = BackEnd::Memory::Host ;
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
    }
    /* setup AMG on GPUs */
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
    /* use hypre's SpGEMM instead of cuSPARSE */
    HYPRE_SetSpGemmUseCusparse(true);
    /* use GPU RNG */
    HYPRE_SetUseGpuRand(true);
    bool useHypreGpuMemPool = false ;
    bool useUmpireGpuMemPool = false ;
    if (useHypreGpuMemPool) {
      /* use hypre's GPU memory pool */
      //HYPRE_SetGPUMemoryPoolSize(bin_growth, min_bin, max_bin, max_bytes);
    }
    else if (useUmpireGpuMemPool) {
       /* or use Umpire GPU memory pool */
       //HYPRE_SetUmpireUMPoolName("HYPRE_UM_POOL_TEST");
       //HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE_POOL_TEST");
     }
  }
  else
  {
    m_memory_type = BackEnd::Memory::Host ;
    m_exec_space = BackEnd::Exec::Host ;
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
  }
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

void
HypreInternalLinearSolver::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  m_parallel_mng = pm;
}

/*---------------------------------------------------------------------------*/
void
HypreInternalLinearSolver::initializeLibrary(bool exec_on_device,
                                             bool use_device_memory,
                                             Integer device_id)
{
  if(Alien::HypreInternalLinearSolver::m_library_plugin_is_initialized) return ;
  HypreInternalLinearSolver::m_library_plugin.reset(new HypreLibrary(exec_on_device,use_device_memory,device_id)) ;
}

void
HypreInternalLinearSolver::init()
{
  if(HypreInternalLinearSolver::m_library_plugin_is_initialized) return ;
#ifdef HYPRE_USING_CUDA
  if(m_options->execSpace() == HypreOptionTypes::Device)
  {
      if(m_options->memoryType() == HypreOptionTypes::DeviceMemory)
        HypreInternalLinearSolver::m_library_plugin.reset(new HypreLibrary(true,true,m_gpu_device_id)) ;
      else
        HypreInternalLinearSolver::m_library_plugin.reset(new HypreLibrary(true,false)) ;
      alien_info([&] {
        cout()<<"Hypre Initialisation : Exec on Device ";
        switch(m_options->memoryType())
        {
          case HypreOptionTypes::HostMemory:
            cout()<<"                       use Host Memory";
            break ;
          case HypreOptionTypes::DeviceMemory:
            cout()<<"                       use Device Memory";
            break ;
          case HypreOptionTypes::ShareMemory:
            cout()<<"                       use Share Memory";
            break ;
          default:
            cout()<<"                       use Host Memory";
        }
      });
   }
   else
#endif
   {
     HypreInternalLinearSolver::m_library_plugin.reset(new HypreLibrary(false,false)) ;
     alien_info([&] {
       cout()<<"Hypre Initialisation : Exec on Host ";
       cout()<<"                       use Host Memory";
     });
   }
   HypreInternalLinearSolver::m_library_plugin_is_initialized = true ;
}

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
struct HypreInternalLinearSolver::Impl
: public ObjectWithTrace
{
  int output_level = 0 ;
  std::string solver_name = "undefined";
  std::string precond_name = "undefined";

  HYPRE_Solver solver = nullptr;
  HYPRE_Solver preconditioner = nullptr;

  // acces aux fonctions du preconditionneur
  HYPRE_PtrToParSolverFcn precond_solve_function = nullptr;
  HYPRE_PtrToParSolverFcn precond_setup_function = nullptr;
  int (*precond_destroy_function)(HYPRE_Solver) = nullptr;

  int (*solver_set_print_level_function)(HYPRE_Solver, int) = nullptr;
  int (*solver_set_tol_function)(HYPRE_Solver, double) = nullptr;
  int (*solver_set_precond_function)(HYPRE_Solver, HYPRE_PtrToParSolverFcn,
      HYPRE_PtrToParSolverFcn, HYPRE_Solver) = nullptr;
  int (*solver_setup_function)(
      HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector) = nullptr;
  int (*solver_solve_function)(
      HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector) = nullptr;
  int (*solver_get_num_iterations_function)(HYPRE_Solver, int*) = nullptr;
  int (*solver_get_final_relative_residual_function)(HYPRE_Solver, double*) = nullptr;
  int (*solver_destroy_function)(HYPRE_Solver) = nullptr;


  HYPRE_ParCSRMatrix par_a;
  HYPRE_ParVector    par_rhs;
  HYPRE_ParVector    par_x;

  bool m_initialized = false ;
  bool m_is_setup = false ;

  void checkError(const Arccore::String& msg, int ierr, int skipError = 0) const;
  void init(IOptionsHypreSolver* options, MPI_Comm comm);
  void setUp(const HYPRE_IJMatrix& ij_matrix);
  void setUp(const HYPRE_IJVector& ij_b,
             HYPRE_IJVector& ij_x);
  bool solve();
  void getStatus(Status& status);
  void end();
} ;

void
HypreInternalLinearSolver::Impl::checkError(
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


void
HypreInternalLinearSolver::Impl::init(IOptionsHypreSolver* options, MPI_Comm comm)
{
  using namespace Alien;
  using namespace Alien::Internal;

  int output_level = options->outputLevel() ;
  if(options->verbose())
    output_level = std::min(1,output_level);

  switch (options->preconditioner())
  {
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
      /*
      !  0 -> CLJP-coarsening (a parallel coarsening algorithm using independent sets
      !  1 -> classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
      !  3 -> classical Ruge-Stueben coarsening on each processor, followed by a third pass,
      !       which adds coarse points on the boundaries
      !  6 -> Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points
#if !defined(USE_HYPREV8) && !defined(USE_HYPREV10)
      !  7 -> CLJP-coarsening (using a fixed random vector, for debugging purposes only)
      !  8 -> PMIS-coarsening (a parallel coarsening algorithm using independent sets,
      !       generating lower complexities than CLJP, might also lead to slower convergence)
      !  9 -> PMIS-coarsening (using a fixed random vector, for debugging purposes only)
      !  10-> HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently,
      !       followed by PMIS using the interior C-points generated as its first independent set)
      !  11-> one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
       */
      int coarsening_opt = 8 ;
      if(options->amgCoarsenType()=="PMIS")
        coarsening_opt = 8 ;
      if(options->amgCoarsenType()=="CLJP")
        coarsening_opt = 0 ;
      if(options->amgCoarsenType()=="Ruge-Stueben")
        coarsening_opt = 1 ;
      if(options->amgCoarsenType()=="Falgout")
        coarsening_opt = 6 ;
      if(options->amgCoarsenType()=="HMIS")
        coarsening_opt = 10 ;
      if(options->amgCoarsenType()=="One-Pass-Ruge-Stueben")
        coarsening_opt = 11 ;

      /*
             ! Type of Interpolation
      ! 0-> Interpolation = modified classical interpolation
      ! 1-> Interpolation = LS interpolation
      ! 2-> Interpolation = modified classical interpolation for hyperbolic PDEs
      ! 3-> Interpolation = direct interpolation with separation of weights
      ! 4-> Interpolation = multipass interpolation
      ! 5-> Interpolation = multipass interpolation with separation of weights
      ! 6-> Interpolation = extended interpolation
      ! 7-> Interpolation = extended interpolation (if no common C point)
      ! 8-> Interpolation = standard interpolation
      ! 9-> Interpolation = standard interpolation with separation of weights
      ! 10->Interpolation = block classical interpolation for nodal systems AMG
      ! 11->Interpolation = block classical interpolation with diagonal blocks for nodal systems AMG
      ! 12-> Interpolation = F-F interpolation
      ! 13-> Interpolation = F-F1 interpolation
      ! -1 : default parameter of Hypre : 0
        - classical
        - direct
        - multipass
        - multipass-wts
        - ext+i
        - ext+i-cc
        - standard
        - standard-wts
        - FF
        - FF1
       */
      int interpolation_type = 7;
      if(options->amgInterpType()=="classical")
        interpolation_type = 0 ;
      if(options->amgInterpType()=="ls-interpolation")
        interpolation_type = 1 ;
      if(options->amgInterpType()=="classical-hyperbolic-pde")
        interpolation_type = 2 ;
      if(options->amgInterpType()=="direct")
        interpolation_type = 3 ;
      if(options->amgInterpType()=="multipass")
        interpolation_type = 4 ;
      if(options->amgInterpType()=="multipass-wts")
        interpolation_type = 5 ;
      if(options->amgInterpType()=="ext+i")
        interpolation_type = 6 ;
      if(options->amgInterpType()=="ext+i-cc")
        interpolation_type = 7;
      if(options->amgInterpType()=="standard")
        interpolation_type = 10;
      if(options->amgInterpType()=="standard-wts")
        interpolation_type = 11;
      if(options->amgInterpType()=="FF")
        interpolation_type = 12;
      if(options->amgInterpType()=="FF1")
        interpolation_type = 13;

      double strong_threshold = options->amgStrongThreshold() ;
      int max_levels = options->amgMaxLevels() ;
      double max_row_sum = options->amgMaxRowSum() ;
      int ierr = 0;
      ierr = HYPRE_BoomerAMGSetMaxIter(preconditioner,1) ;
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
        alien_fatal([&] {
          cout() << "Error while calling HYPRE_BoomerAMGSetMaxIter with default value";
        });
      }

      ierr = HYPRE_BoomerAMGSetTol(preconditioner,1.e-7) ;

      ierr = HYPRE_BoomerAMGSetMaxLevels(preconditioner,max_levels) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetMaxLevels with default value";
        });
      }
      ierr = HYPRE_BoomerAMGSetMaxRowSum(preconditioner,max_row_sum) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
                  cout() << "Error while calling HYPRE_BoomerAMGSetMaxRowSum with default value";
        });
      }

      ierr = HYPRE_BoomerAMGSetCycleType(preconditioner,1) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
             cout() << "Error while calling HYPRE_BoomerAMGSetCycleType with default value";
        });
      }
      /*
     ! Type of Smoother
     ! 0 -> weigted jacobi
     ! 1 -> sequentiel Gauss seidel
     ! 3 -> Gauss Seidel Jacobi
     ! 6 -> symetric Gauss Seidel
     ! 9 -> Gauss Elimination
     ! -1 : default parameter : 3
     Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination
       */
      int relax_type = 3 ;
      if(options->amgRelaxType()=="Jacobi")
        relax_type = 0 ;
      if(options->amgRelaxType()=="sequential-Gauss-Seidel")
        relax_type = 1 ;
      if(options->amgRelaxType()=="backward-Gauss-Seidel")
        relax_type = 2 ;
      if(options->amgRelaxType()=="hybrid-Gauss-Seidel")
        relax_type = 3 ;
      if(options->amgRelaxType()=="backward-hybrid-Gauss-Seidel")
          relax_type = 4 ;
      if(options->amgRelaxType()=="symetric-hybrid-Gauss-Seidel")
        relax_type = 5 ;
      if(options->amgRelaxType()=="l1-hybrid-Gauss-Seidel")
        relax_type = 6 ;
      if(options->amgRelaxType()=="backward-l1-hybrid-Gauss-Seidel")
        relax_type = 7 ;
      if(options->amgRelaxType()=="l1-hybrid-Gauss-Seidel")
        relax_type = 8 ;
      if(options->amgRelaxType()=="l1-Gauss-Seidel")
        relax_type = 9 ;
      if(options->amgRelaxType()=="forward-l1-Gauss-Seidel")
        relax_type = 13 ;
      if(options->amgRelaxType()=="backward-l1-Gauss-Seidel")
        relax_type = 14 ;
      if(options->amgRelaxType()=="CG")
        relax_type = 15 ;
      if(options->amgRelaxType()=="Chebyshev")
        relax_type = 16 ;
      if(options->amgRelaxType()=="l1-hybrid-Gauss-Seidel-v2")
        relax_type = 18 ;
      ierr = HYPRE_BoomerAMGSetRelaxType(preconditioner,relax_type) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetRelaxType : "<<options->amgRelaxType()<<" IERR="<<ierr;
        });
      }

       ierr = HYPRE_BoomerAMGSetCoarsenType(preconditioner,coarsening_opt) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
             cout() << "Error while calling HYPRE_BoomerAMGSetCoarsenType";
         });
       }

       ierr = HYPRE_BoomerAMGSetNumSweeps(preconditioner,1) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetNumSweeps";
         });
       }

       ierr = HYPRE_BoomerAMGSetSmoothNumLevels(preconditioner,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetSmoothNumLevels";
         });
       }

       ierr = HYPRE_BoomerAMGSetMeasureType(preconditioner,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetMeasureType";
         });
       }

       ierr = HYPRE_BoomerAMGSetAggNumLevels(preconditioner,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetAggNumLevels";
         });
       }

       ierr = HYPRE_BoomerAMGSetNumPaths(preconditioner,1) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetNumPaths";
         });
       }

       ierr = HYPRE_BoomerAMGSetInterpType(preconditioner,interpolation_type) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetInterpType";
         });
       }

       ierr = HYPRE_BoomerAMGSetSmoothNumLevels(preconditioner,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetSmoothNumLevels";
         });
       }

       ierr = HYPRE_BoomerAMGSetSmoothType(preconditioner,relax_type) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetSmoothType";
         });
       }

       ierr = HYPRE_BoomerAMGSetStrongThreshold(preconditioner,strong_threshold) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetStrongThreshold";
         });
       }

       ierr = HYPRE_BoomerAMGSetMeasureType(preconditioner,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetMeasureType with default value";
         });
       }
       ierr = HYPRE_BoomerAMGSetAggNumLevels(preconditioner,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetAggNumLevels with default value";
         });
       }
       ierr = HYPRE_BoomerAMGSetNumPaths(preconditioner,1) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetNumPaths with default value";
         });
       }
       checkError("Hypre " + precond_name + " solver Setlogging",
                   HYPRE_BoomerAMGSetLogging(preconditioner,output_level));
       checkError("Hypre " + precond_name + " solver SetPrintLevel",
                   HYPRE_BoomerAMGSetPrintLevel(preconditioner, output_level));
    }
    break;
  case HypreOptionTypes::ParaSailsPC:
    {
      precond_name = "parasails";
      checkError(
          "Hypre ParaSails preconditioner", HYPRE_ParaSailsCreate(comm, &preconditioner));
      precond_solve_function = HYPRE_ParaSailsSolve;
      precond_setup_function = HYPRE_ParaSailsSetup;
      precond_destroy_function = HYPRE_ParaSailsDestroy;
    }
    break;
  case HypreOptionTypes::EuclidPC:
    {
      precond_name = "euclid";
      checkError("Hypre Euclid preconditioner", HYPRE_EuclidCreate(comm, &preconditioner));
      precond_solve_function = HYPRE_EuclidSolve;
      precond_setup_function = HYPRE_EuclidSetup;
      precond_destroy_function = HYPRE_EuclidDestroy;
    }
    break;
#if HYPRE_RELEASE_NUMBER >= 22200
  case HypreOptionTypes::BJILUKPC:
    {
      precond_name = "iluk";
      checkError("Hypre ILU preconditioner", HYPRE_ILUCreate(&preconditioner));
      /* (Recommended) General solver options */
      int ilu_type = 0 ;
      int max_iter = 1 ;
      double tol = 0. ;
      int reordering = 0 ;
      int fill = options->ilukLevel() ;
      int print_level = 3 ;
      checkError("Hypre ILUK preconditioner Type SetUp    ", HYPRE_ILUSetType(preconditioner, ilu_type)); /* 0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50 */
      checkError("Hypre ILUK preconditioner Max Iter      ", HYPRE_ILUSetMaxIter(preconditioner, max_iter));
      checkError("Hypre ILUK preconditioner Tol           ", HYPRE_ILUSetTol(preconditioner, tol));
      checkError("Hypre ILUK preconditioner Reodering     ", HYPRE_ILUSetLocalReordering(preconditioner, reordering)); /* 0: none, 1: RCM */
      checkError("Hypre ILUK preconditioner Fill In Level ", HYPRE_ILUSetLevelOfFill(preconditioner, fill));
      if (output_level > 2) {
          checkError("Hypre ILUK preconditioner PrintLevel SetUp", HYPRE_ILUSetPrintLevel(preconditioner, print_level));
      }
      precond_solve_function = HYPRE_ILUSolve;
      precond_setup_function = HYPRE_ILUSetup;
      precond_destroy_function = HYPRE_ILUDestroy;
    }
    break;
  case HypreOptionTypes::BJILUTPC:
    {
      precond_name = "ilut";
      checkError("Hypre ILU preconditioner", HYPRE_ILUCreate(&preconditioner));
      /* (Recommended) General solver options */
      int ilu_type = 1 ;
      int max_iter = 1 ;
      double tol = 0. ;
      int reordering = 0 ;
      double threshold = options->ilutThreshold() ;
      int max_nnz_row = options->ilutMaxNnz() ;
      int print_level = 3 ;
      checkError("Hypre ILUT preconditioner Type SetUp      ", HYPRE_ILUSetType(preconditioner, ilu_type)); /* 0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50 */
      checkError("Hypre ILUT preconditioner Max Iter        ", HYPRE_ILUSetMaxIter(preconditioner, max_iter));
      checkError("Hypre ILUT preconditioner Tol             ", HYPRE_ILUSetTol(preconditioner, tol));
      checkError("Hypre ILUT preconditioner Reodering       ", HYPRE_ILUSetLocalReordering(preconditioner, reordering)); /* 0: none, 1: RCM */
      if(max_nnz_row>0)
        checkError("Hypre ILUT preconditioner Max Nnz per Row ", HYPRE_ILUSetMaxNnzPerRow(preconditioner, max_nnz_row));
      if(threshold>0.)
        checkError("Hypre ILUT preconditioner Threshold       ", HYPRE_ILUSetDropThreshold(preconditioner, threshold));
      if (output_level > 2) {
          checkError("Hypre ILU preconditioner PrintLevel SetUp", HYPRE_ILUSetPrintLevel(preconditioner, print_level));
      }
      precond_solve_function = HYPRE_ILUSolve;
      precond_setup_function = HYPRE_ILUSetup;
      precond_destroy_function = HYPRE_ILUDestroy;
    }
    break;
  case HypreOptionTypes::FSAIPC:
    {
      precond_name = "fsai";
      int max_steps = 5 ;
      int max_step_size = 3;
      double kap_tolerance = 1.e-3 ;
      checkError("Hypre FSAI preconditioner", HYPRE_FSAICreate(&preconditioner));
      checkError("Hypre FSAI preconditioner", HYPRE_FSAISetMaxSteps(preconditioner, max_steps));
      checkError("Hypre FSAI preconditioner", HYPRE_FSAISetMaxStepSize(preconditioner, max_step_size));
      checkError("Hypre FSAI preconditioner", HYPRE_FSAISetKapTolerance(preconditioner, kap_tolerance));
      precond_solve_function = HYPRE_FSAISolve;
      precond_setup_function = HYPRE_FSAISetup;
      precond_destroy_function = HYPRE_FSAIDestroy;
    }
    break;
#endif
  default:
    alien_fatal([&] { cout() << "Undefined Hypre preconditioner option"; });
    break;
  }

  int max_it = options->numIterationsMax();
  double rtol = options->stopCriteriaValue();

  switch (options->solver())
  {
  case HypreOptionTypes::AMG:
    {
      solver_name = "amg";
      checkError("Hypre AMG solver", HYPRE_BoomerAMGCreate(&solver));

      /*
      !  0 -> CLJP-coarsening (a parallel coarsening algorithm using independent sets
      !  1 -> classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
      !  3 -> classical Ruge-Stueben coarsening on each processor, followed by a third pass,
      !       which adds coarse points on the boundaries
      !  6 -> Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points
  #if !defined(USE_HYPREV8) && !defined(USE_HYPREV10)
      !  7 -> CLJP-coarsening (using a fixed random vector, for debugging purposes only)
      !  8 -> PMIS-coarsening (a parallel coarsening algorithm using independent sets,
      !       generating lower complexities than CLJP, might also lead to slower convergence)
      !  9 -> PMIS-coarsening (using a fixed random vector, for debugging purposes only)
      !  10-> HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently,
      !       followed by PMIS using the interior C-points generated as its first independent set)
      !  11-> one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
       */
      int coarsening_opt = 8 ;
      if(options->amgCoarsenType()=="PMIS")
        coarsening_opt = 8 ;
      if(options->amgCoarsenType()=="CLJP")
        coarsening_opt = 0 ;
      if(options->amgCoarsenType()=="Ruge-Stueben")
        coarsening_opt = 1 ;
      if(options->amgCoarsenType()=="Falgout")
        coarsening_opt = 6 ;
      if(options->amgCoarsenType()=="HMIS")
        coarsening_opt = 10 ;
      if(options->amgCoarsenType()=="One-Pass-Ruge-Stueben")
        coarsening_opt = 11 ;

      /*
             ! Type of Interpolation
      ! 0-> Interpolation = modified classical interpolation
      ! 1-> Interpolation = LS interpolation
      ! 2-> Interpolation = modified classical interpolation for hyperbolic PDEs
      ! 3-> Interpolation = direct interpolation with separation of weights
      ! 4-> Interpolation = multipass interpolation
      ! 5-> Interpolation = multipass interpolation with separation of weights
      ! 6-> Interpolation = extended interpolation
      ! 7-> Interpolation = extended interpolation (if no common C point)
      ! 8-> Interpolation = standard interpolation
      ! 9-> Interpolation = standard interpolation with separation of weights
      ! 10->Interpolation = block classical interpolation for nodal systems AMG
      ! 11->Interpolation = block classical interpolation with diagonal blocks for nodal systems AMG
      ! 12-> Interpolation = F-F interpolation
      ! 13-> Interpolation = F-F1 interpolation
      ! -1 : default parameter of Hypre : 0
        - classical
        - direct
        - multipass
        - multipass-wts
        - ext+i
        - ext+i-cc
        - standard
        - standard-wts
        - FF
        - FF1
       */
      int interpolation_type = 7;
      if(options->amgInterpType()=="classical")
        interpolation_type = 0 ;
      if(options->amgInterpType()=="ls-interpolation")
        interpolation_type = 1 ;
      if(options->amgInterpType()=="classical-hyperbolic-pde")
        interpolation_type = 2 ;
      if(options->amgInterpType()=="direct")
        interpolation_type = 3 ;
      if(options->amgInterpType()=="multipass")
        interpolation_type = 4 ;
      if(options->amgInterpType()=="multipass-wts")
        interpolation_type = 5 ;
      if(options->amgInterpType()=="ext+i")
        interpolation_type = 6 ;
      if(options->amgInterpType()=="ext+i-cc")
        interpolation_type = 7;
      if(options->amgInterpType()=="standard")
        interpolation_type = 10;
      if(options->amgInterpType()=="standard-wts")
        interpolation_type = 11;
      if(options->amgInterpType()=="FF")
        interpolation_type = 12;
      if(options->amgInterpType()=="FF1")
        interpolation_type = 13;

      double strong_threshold = options->amgStrongThreshold() ;
      int max_levels = options->amgMaxLevels() ;
      double max_row_sum = options->amgMaxRowSum() ;
      int ierr = 0;
      ierr = HYPRE_BoomerAMGSetMaxLevels(solver,max_levels) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetMaxLevels with default value";
        });
      }
      ierr = HYPRE_BoomerAMGSetMaxRowSum(solver,max_row_sum) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
                  cout() << "Error while calling HYPRE_BoomerAMGSetMaxRowSum with default value";
        });
      }

      ierr = HYPRE_BoomerAMGSetCycleType(solver,1) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
             cout() << "Error while calling HYPRE_BoomerAMGSetCycleType with default value";
        });
      }
      /*
     ! Type of Smoother
     ! 0 -> weigted jacobi
     ! 1 -> sequentiel Gauss seidel
     ! 3 -> Gauss Seidel Jacobi
     ! 6 -> symetric Gauss Seidel
     ! 9 -> Gauss Elimination
     ! -1 : default parameter : 3
     Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination
       */
      int relax_type = 3 ;
      if(options->amgRelaxType()=="Jacobi")
        relax_type = 0 ;
      if(options->amgRelaxType()=="sequential-Gauss-Seidel")
        relax_type = 1 ;
      if(options->amgRelaxType()=="backward-Gauss-Seidel")
        relax_type = 2 ;
      if(options->amgRelaxType()=="hybrid-Gauss-Seidel")
        relax_type = 3 ;
      if(options->amgRelaxType()=="backward-hybrid-Gauss-Seidel")
          relax_type = 4 ;
      if(options->amgRelaxType()=="symetric-hybrid-Gauss-Seidel")
        relax_type = 5 ;
      if(options->amgRelaxType()=="l1-hybrid-Gauss-Seidel")
        relax_type = 6 ;
      if(options->amgRelaxType()=="backward-l1-hybrid-Gauss-Seidel")
        relax_type = 7 ;
      if(options->amgRelaxType()=="l1-hybrid-Gauss-Seidel")
        relax_type = 8 ;
      if(options->amgRelaxType()=="l1-Gauss-Seidel")
        relax_type = 9 ;
      if(options->amgRelaxType()=="forward-l1-Gauss-Seidel")
        relax_type = 13 ;
      if(options->amgRelaxType()=="backward-l1-Gauss-Seidel")
        relax_type = 14 ;
      if(options->amgRelaxType()=="CG")
        relax_type = 15 ;
      if(options->amgRelaxType()=="Chebyshev")
        relax_type = 16 ;
      if(options->amgRelaxType()=="l1-hybrid-Gauss-Seidel-v2")
        relax_type = 18 ;
      ierr = HYPRE_BoomerAMGSetRelaxType(solver,relax_type) ;
      if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
      if(ierr) {
        alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetRelaxType : "<<options->amgRelaxType()<<" IERR="<<ierr;
        });
      }

       ierr = HYPRE_BoomerAMGSetCoarsenType(solver,coarsening_opt) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
             cout() << "Error while calling HYPRE_BoomerAMGSetCoarsenType";
         });
       }

       ierr = HYPRE_BoomerAMGSetNumSweeps(solver,1) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetNumSweeps";
         });
       }

       ierr = HYPRE_BoomerAMGSetSmoothNumLevels(solver,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetSmoothNumLevels";
         });
       }

       ierr = HYPRE_BoomerAMGSetMeasureType(solver,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetMeasureType";
         });
       }

       ierr = HYPRE_BoomerAMGSetAggNumLevels(solver,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetAggNumLevels";
         });
       }

       ierr = HYPRE_BoomerAMGSetNumPaths(solver,1) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetNumPaths";
         });
       }

       ierr = HYPRE_BoomerAMGSetInterpType(solver,interpolation_type) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetInterpType";
         });
       }

       ierr = HYPRE_BoomerAMGSetSmoothNumLevels(solver,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetSmoothNumLevels";
         });
       }

       ierr = HYPRE_BoomerAMGSetSmoothType(solver,relax_type) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetSmoothType";
         });
       }

       ierr = HYPRE_BoomerAMGSetStrongThreshold(solver,strong_threshold) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetStrongThreshold";
         });
       }

       ierr = HYPRE_BoomerAMGSetMeasureType(solver,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetMeasureType with default value";
         });
       }
       ierr = HYPRE_BoomerAMGSetAggNumLevels(solver,0) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
            cout() << "Error while calling HYPRE_BoomerAMGSetAggNumLevels with default value";
         });
       }
       ierr = HYPRE_BoomerAMGSetNumPaths(solver,1) ;
       if( ierr == HYPRE_ERROR_CONV) ierr = 0 ;
       if(ierr) {
         alien_fatal([&] {
           cout() << "Error while calling HYPRE_BoomerAMGSetNumPaths with default value";
         });
       }
       checkError("Hypre " + precond_name + " solver Setlogging",
                   HYPRE_BoomerAMGSetLogging(solver,output_level));
       checkError("Hypre " + precond_name + " solver SetPrintLevel",
                   HYPRE_BoomerAMGSetPrintLevel(solver, output_level));

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
    }
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
  m_initialized = true ;
}



void
HypreInternalLinearSolver::Impl::setUp(const HYPRE_IJMatrix& ij_matrix)
{
  checkError(
      "Hypre Matrix GetObject", HYPRE_IJMatrixGetObject(ij_matrix, (void**)&par_a));
}

void
HypreInternalLinearSolver::Impl::setUp(const HYPRE_IJVector& bij_vector,
                                        HYPRE_IJVector& xij_vector)
{
  checkError("Hypre RHS Vector GetObject",
      HYPRE_IJVectorGetObject(bij_vector, (void**)&par_rhs));
  checkError("Hypre Unknown Vector GetObject",
      HYPRE_IJVectorGetObject(xij_vector, (void**)&par_x));

  checkError("Hypre " + solver_name + " solver Setup",
      (*solver_setup_function)(solver, par_a, par_rhs, par_x));
  m_is_setup = true ;
}

bool
HypreInternalLinearSolver::Impl::solve()
{
  assert(m_is_setup) ;
  int code = (*solver_solve_function)(solver, par_a, par_rhs, par_x) ;
  return (code==0 || code==HYPRE_ERROR_CONV);
}

void HypreInternalLinearSolver::Impl::getStatus(Status& status)
{
   checkError("Hypre " + solver_name + " solver GetNumIterations",
      (*solver_get_num_iterations_function)(solver, &status.iteration_count));
  checkError("Hypre " + solver_name + " solver GetFinalResidual",
      (*solver_get_final_relative_residual_function)(solver, &status.residual));
}

void
HypreInternalLinearSolver::Impl::end()
{
  checkError(
      "Hypre " + solver_name + " solver Destroy", (*solver_destroy_function)(solver));
  if (precond_destroy_function)
    checkError("Hypre " + precond_name + " preconditioner Destroy",
        (*precond_destroy_function)(preconditioner));
}

void
HypreInternalLinearSolver::init(const HypreMatrix& A)
{
  const HYPRE_IJMatrix& ij_matrix = A.internal()->internal();
  HYPRE_ClearAllErrors() ;

  auto pm = m_parallel_mng->communicator();
  MPI_Comm comm = (pm.isValid())
      ? static_cast<const MPI_Comm>(pm)
      : MPI_COMM_WORLD;

  if(m_impl.get())
  {
    m_impl->end() ;
  }
  m_impl.reset(new Impl);
  m_impl->init(m_options,comm);
  m_impl->setUp(ij_matrix);
}

bool
HypreInternalLinearSolver::solve(const HypreMatrix& A,
                                 const HypreVector& b,
                                 HypreVector& x)
{
  const HYPRE_IJMatrix& ij_matrix = A.internal()->internal();
  const HYPRE_IJVector& bij_vector = b.internal()->internal();
  HYPRE_IJVector& xij_vector = x.internal()->internal();
  init(A) ;
  m_impl->setUp(ij_matrix) ;
  m_impl->setUp(bij_vector,xij_vector) ;
  m_status.succeeded = m_impl->solve() ;
  m_impl->getStatus(m_status) ;
  m_impl->end() ;
  m_impl.reset() ;
  return m_status.succeeded;

}

bool
HypreInternalLinearSolver::solve(const HypreVector& b,
                                 HypreVector& x)
{
  const HYPRE_IJVector& bij_vector = b.internal()->internal();
  HYPRE_IJVector& xij_vector = x.internal()->internal();
  assert(m_impl.get()!=nullptr) ;
  assert(m_impl->m_initialized) ;
  if(!m_impl->m_is_setup)
    m_impl->setUp(bij_vector,xij_vector) ;
  m_status.succeeded = m_impl->solve() ;
  return m_status.succeeded;
}

void
HypreInternalLinearSolver::end()
{
  if(m_impl.get())
  {
    m_impl->end();
    m_impl.reset();
  }
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
