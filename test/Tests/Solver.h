#ifndef TESTS_REFSEMANTICMVHANDLERS_SOLVERCONFIGURATION_H
#define TESTS_REFSEMANTICMVHANDLERS_SOLVERCONFIGURATION_H

#include <memory>

#include <boost/program_options/variables_map.hpp>

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/utils/parameter_manager/BaseParameterManager.h>

#include <ALIEN/AlienExternalPackages.h>
#include <ALIEN/Alien-IFPENSolvers.h>

#ifdef ALIEN_USE_MTL4
#include <ALIEN/Kernels/MTL/linear_solver/arcane/MTLLinearSolverService.h>
#include <ALIEN/Kernels/MTL/linear_solver/MTLOptionTypes.h>
#include <ALIEN/axl/MTLLinearSolver_IOptions.h>
#include <ALIEN/axl/MTLLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_PETSC
#include <ALIEN/Kernels/PETSc/algebra/PETScLinearAlgebra.h>
#include <ALIEN/Kernels/PETSc/linear_solver/arcane/PETScLinearSolverService.h>
#include <ALIEN/Kernels/PETSc/linear_solver/PETScInternalLinearSolver.h>
// preconditionner
#include <ALIEN/Kernels/PETSc/linear_solver/arcane/PETScPrecConfigDiagonalService.h>
#include <ALIEN/Kernels/PETSc/linear_solver/arcane/PETScPrecConfigJacobiService.h>
#include <ALIEN/Kernels/PETSc/linear_solver/arcane/PETScPrecConfigNoPreconditionerService.h>
#include <ALIEN/axl/PETScPrecConfigDiagonal_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigDiagonal_StrongOptions.h>
#include <ALIEN/axl/PETScPrecConfigJacobi_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigJacobi_StrongOptions.h>
#include <ALIEN/axl/PETScPrecConfigNoPreconditioner_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigNoPreconditioner_StrongOptions.h>
// solver
#include <ALIEN/Kernels/PETSc/linear_solver/arcane/PETScSolverConfigBiCGStabService.h>
#include <ALIEN/Kernels/PETSc/linear_solver/arcane/PETScSolverConfigLUService.h>
#include <ALIEN/Kernels/PETSc/linear_solver/IPETScKSP.h>
#include <ALIEN/Kernels/PETSc/linear_solver/IPETScPC.h>
#include <ALIEN/axl/PETScSolverConfigBiCGStab_IOptions.h>
#include <ALIEN/axl/PETScSolverConfigBiCGStab_StrongOptions.h>
#include <ALIEN/axl/PETScSolverConfigLU_IOptions.h>
#include <ALIEN/axl/PETScSolverConfigLU_StrongOptions.h>
// root linear solver instance
#include <ALIEN/axl/PETScLinearSolver_IOptions.h>
#include <ALIEN/axl/PETScLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_HYPRE
#include <ALIEN/Kernels/Hypre/linear_solver/HypreOptionTypes.h>
#include <ALIEN/Kernels/Hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/axl/HypreSolver_IOptions.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_IFPSOLVER
#include <ALIEN/Kernels/ifp/linear_solver/arcane/IFPLinearSolverService.h>
#include <ALIEN/Kernels/ifp/linear_solver/IFPSolverProperty.h>
#include <ALIEN/axl/IFPLinearSolver_IOptions.h>
#include <ALIEN/axl/IFPLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_MCGSOLVER
#include <ALIEN/Kernels/mcg/linear_solver/arcane/GPULinearSolver.h>
#include <ALIEN/Kernels/mcg/linear_solver/GPUOptionTypes.h>
#include <ALIEN/axl/GPUSolver_IOptions.h>
#include <ALIEN/axl/GPUSolver_StrongOptions.h>
#endif

#include <Tests/Environment.h>

namespace Environment {

extern std::shared_ptr<Alien::ILinearSolver> createSolver(boost::program_options::variables_map& vm)
{
  auto* pm = Environment::parallelMng();
  auto* tm = Environment::traceMng();
  
  std::string solver_package = vm["solver-package"].as<std::string>();
  
  tm->info() << "Try to create solver-package : " << solver_package;
  double tol = vm["tol"].as<double>();
  int max_iter = vm["max-iter"].as<int>();
  if(solver_package.compare("petsc") == 0)
  {
#ifdef ALIEN_USE_PETSC
    std::shared_ptr<Alien::IPETScPC> prec = nullptr;
    // preconditionner service
    std::string precond_type_s = vm["precond"].as<std::string>();
    if(precond_type_s.compare("bjacobi")==0){
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigJacobi>();
      prec = std::make_shared<Alien::PETScPrecConfigJacobiService>(pm, options_prec);
    }
    else if(precond_type_s.compare("diag")==0){
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigDiagonal>();
      prec = std::make_shared<Alien::PETScPrecConfigDiagonalService>(pm, options_prec);
    }
    else if(precond_type_s.compare("none")==0){
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigNoPreconditioner>();
      prec = std::make_shared<Alien::PETScPrecConfigNoPreconditionerService>(pm, options_prec);
    } 
    std::string solver = vm["solver"].as<std::string>();
    if(solver.compare("bicgs")==0)
    {
      // solver service bicgs
      using namespace PETScSolverConfigBiCGStabOptionsNames;
      auto options_solver = std::make_shared<StrongOptionsPETScSolverConfigBiCGStab>(
										     _numIterationsMax = max_iter,
										     _stopCriteriaValue = tol,
										     _preconditioner = prec
										     );
      // root petsc option
      auto root_options = std::make_shared<StrongOptionsPETScLinearSolver>(
									   PETScLinearSolverOptionsNames::_solver = std::make_shared<Alien::PETScSolverConfigBiCGStabService>(pm, options_solver)
									   );
      // root petsc service
      return std::make_shared<Alien::PETScLinearSolverService>(pm, root_options);
    }
    if(solver.compare("lu")==0)
    {
      // solver service lu
      using namespace PETScSolverConfigLUOptionsNames;
      auto options_solver = std::make_shared<StrongOptionsPETScSolverConfigLU>();
      // root petsc option
      auto root_options = std::make_shared<StrongOptionsPETScLinearSolver>(
									   PETScLinearSolverOptionsNames::_solver = std::make_shared<Alien::PETScSolverConfigLUService>(pm, options_solver)
									   );
      // root petsc service
      return std::make_shared<Alien::PETScLinearSolverService>(pm, root_options);
    }    
    tm->fatal() << "*** solver " << solver << " not available in test!";
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }
	
  if(solver_package.compare("hypre") == 0)
  {
#ifdef ALIEN_USE_HYPRE
    std::string solver_type_s = vm["solver"].as<std::string>();
    HypreOptionTypes::eSolver solver_type = OptionsHypreSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = vm["precond"].as<std::string>();
    HypreOptionTypes::ePreconditioner precond_type = OptionsHypreSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    using namespace HypreSolverOptionsNames;
    auto options = std::make_shared<StrongOptionsHypreSolver>(
							      _numIterationsMax = max_iter,
							      _stopCriteriaValue = tol,
							      _solver = solver_type,
							      _preconditioner = precond_type
							      );
    // service
    return std::make_shared<Alien::HypreLinearSolver>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }
	
  if(solver_package.compare("ifpsolver") == 0)
  {
#ifdef ALIEN_USE_IFPSOLVER
    std::string precond_type_s = vm["precond"].as<std::string>();
    IFPSolverProperty::ePrecondType precond_type = OptionsIFPLinearSolverUtils::stringToPrecondOptionEnum(precond_type_s);
    // options
    auto options = std::make_shared<StrongOptionsIFPLinearSolver>(
								  IFPLinearSolverOptionsNames::_output = vm["output-level"].as<int>(),
								  IFPLinearSolverOptionsNames::_numIterationsMax = max_iter,
								  IFPLinearSolverOptionsNames::_stopCriteriaValue = tol,
								  IFPLinearSolverOptionsNames::_precondOption = precond_type
								  );
    // service
    return std::make_shared<Alien::IFPLinearSolverService>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }
  if(solver_package.compare("mcgsolver") ==0 )
  {
#ifdef ALIEN_USE_MCGSOLVER
    std::string precond_type_s = vm["precond"].as<std::string>();
    GPUOptionTypes::ePreconditioner precond_type = OptionsGPUSolverUtils::stringToPreconditionerEnum(precond_type_s);
    std::string kernel_type_s = vm["kernel"].as<std::string>();
    GPUOptionTypes::eKernelType kernel_type = OptionsGPUSolverUtils::stringToKernelEnum(kernel_type_s);
    // options
    auto options = std::make_shared<StrongOptionsGPUSolver>(
							    GPUSolverOptionsNames::_output = vm["output-level"].as<int>(),
							    GPUSolverOptionsNames::_maxIterationNum = max_iter,
							    GPUSolverOptionsNames::_stopCriteriaValue = tol,
							    GPUSolverOptionsNames::_kernel = kernel_type,
							    GPUSolverOptionsNames::_preconditioner = precond_type
							    );
    // service
    return std::make_shared<Alien::GPULinearSolver>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }
  
  if(solver_package.compare("mtlsolver") ==0 )
  {
#ifdef ALIEN_USE_MTL4
    std::string solver_type_s = vm["solver"].as<std::string>();
    MTLOptionTypes::eSolver solver_type = OptionsMTLLinearSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = vm["precond"].as<std::string>();
    MTLOptionTypes::ePreconditioner precond_type = OptionsMTLLinearSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    auto options = std::make_shared<StrongOptionsMTLLinearSolver>(
								  MTLLinearSolverOptionsNames::_outputLevel = vm["output-level"].as<int>(),
								  MTLLinearSolverOptionsNames::_maxIterationNum = max_iter,
								  MTLLinearSolverOptionsNames::_stopCriteriaValue = tol,
								  MTLLinearSolverOptionsNames::_preconditioner = precond_type,
								  MTLLinearSolverOptionsNames::_solver = solver_type
								  );
    // service
    return std::make_shared<Alien::MTLLinearSolverService>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }
	
  tm->fatal() << "*** package " << solver_package << " not available!";
	
  return std::shared_ptr<Alien::ILinearSolver>();
  }
  
}

#endif
