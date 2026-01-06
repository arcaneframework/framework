// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Solver                                         (C) 2000-2025              */
/*                                                                           */
/* Solver tests                                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef TESTS_REFSEMANTICMVHANDLERS_SOLVERCONFIGURATION_H
#define TESTS_REFSEMANTICMVHANDLERS_SOLVERCONFIGURATION_H

#include <memory>

#include <boost/program_options/variables_map.hpp>

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/utils/parameter_manager/BaseParameterManager.h>

#include <alien/AlienCoreSolvers.h>
#include <alien/kernels/common/linear_solver/arcane/AlienLinearSolver.h>
#include <alien/kernels/common/AlienCoreSolverOptionTypes.h>
#include <ALIEN/axl/AlienCoreSolver_IOptions.h>
#include <ALIEN/axl/AlienCoreSolver_StrongOptions.h>

#ifdef ALIEN_USE_MTL4
#include <alien/AlienExternalPackages.h>
#include <alien/kernels/mtl/linear_solver/arcane/MTLLinearSolverService.h>
#include <alien/kernels/mtl/linear_solver/MTLOptionTypes.h>
#include <ALIEN/axl/MTLLinearSolver_IOptions.h>
#include <ALIEN/axl/MTLLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_PETSC
#include <alien/AlienExternalPackages.h>
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScLinearSolverService.h>
#include <alien/kernels/petsc/linear_solver/PETScInternalLinearSolver.h>
// preconditionner
#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigDiagonalService.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigJacobiService.h>
#include <alien/kernels/petsc/linear_solver/spai/PETScPrecConfigSPAIService.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigNoPreconditionerService.h>
#include <ALIEN/axl/PETScPrecConfigDiagonal_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigDiagonal_StrongOptions.h>
#include <ALIEN/axl/PETScPrecConfigJacobi_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigJacobi_StrongOptions.h>
#include <ALIEN/axl/PETScPrecConfigSPAI_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigSPAI_StrongOptions.h>
#include <ALIEN/axl/PETScPrecConfigNoPreconditioner_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigNoPreconditioner_StrongOptions.h>
// solver
#include <alien/kernels/petsc/linear_solver/arcane/PETScSolverConfigBiCGStabService.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScSolverConfigLUService.h>
#include <alien/kernels/petsc/linear_solver/IPETScKSP.h>
#include <alien/kernels/petsc/linear_solver/IPETScPC.h>
#include <ALIEN/axl/PETScSolverConfigBiCGStab_IOptions.h>
#include <ALIEN/axl/PETScSolverConfigBiCGStab_StrongOptions.h>
#include <ALIEN/axl/PETScSolverConfigLU_IOptions.h>
#include <ALIEN/axl/PETScSolverConfigLU_StrongOptions.h>
#ifdef PETSC_HAVE_MUMPS
#include <alien/kernels/petsc/linear_solver/mumps/PETScSolverConfigMUMPSService.h>
#include <ALIEN/axl/PETScSolverConfigMUMPS_IOptions.h>
#endif
// root linear solver instance
#include <ALIEN/axl/PETScLinearSolver_IOptions.h>
#include <ALIEN/axl/PETScLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_HYPRE
#include <alien/AlienExternalPackages.h>
#include <alien/kernels/hypre/linear_solver/HypreOptionTypes.h>
#include <alien/kernels/hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/axl/HypreSolver_IOptions.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_IFPSOLVER
#include <alien/AlienIFPENSolvers.h>
#include <alien/kernels/ifp/linear_solver/arcane/IFPLinearSolverService.h>
#include <alien/kernels/ifp/linear_solver/IFPSolverProperty.h>
#include <ALIEN/axl/IFPLinearSolver_IOptions.h>
#include <ALIEN/axl/IFPLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_MCGSOLVER
#include <alien/AlienIFPENSolvers.h>
#include <alien/kernels/mcg/linear_solver/arcane/MCGLinearSolver.h>
#include <alien/kernels/mcg/linear_solver/MCGOptionTypes.h>
#include <ALIEN/axl/MCGSolver_IOptions.h>
#include <ALIEN/axl/MCGSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_HTSSOLVER
#include <alien/AlienIFPENSolvers.h>
#include <alien/kernels/hts/linear_solver/HTSOptionTypes.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/hts/linear_solver/HTSInternalLinearSolver.h>
#include <alien/kernels/hts/linear_solver/arcane/HTSLinearSolver.h>
#include <ALIEN/axl/HTSSolver_IOptions.h>
#include <ALIEN/axl/HTSSolver_StrongOptions.h>
#endif

#include <Tests/Environment.h>

namespace Environment {

extern std::shared_ptr<Alien::ILinearSolver>
createSolver(boost::program_options::variables_map& vm)
{
  auto* pm = Environment::parallelMng();
  auto* tm = Environment::traceMng();

  std::string solver_package = vm["solver-package"].as<std::string>();

  tm->info() << "Try to create solver-package : " << solver_package;
  double tol = vm["tol"].as<double>();
  int max_iter = vm["max-iter"].as<int>();

  if (solver_package.compare("alien-core") == 0) {
    std::string solver_type_s = vm["solver"].as<std::string>();
    AlienCoreSolverOptionTypes::eSolver solver_type =
        OptionsAlienCoreSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = vm["precond"].as<std::string>();
    AlienCoreSolverOptionTypes::ePreconditioner precond_type =
        OptionsAlienCoreSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    using namespace AlienCoreSolverOptionsNames;
    auto options = std::make_shared<StrongOptionsAlienCoreSolver>(
        _maxIter = max_iter, _tol = tol, _solver = solver_type,
        _preconditioner = precond_type);
    // service
    return std::make_shared<Alien::AlienLinearSolver>(pm, options);
  }

  if (solver_package.compare("petsc") == 0) {
#ifdef ALIEN_USE_PETSC
    std::shared_ptr<Alien::IPETScPC> prec = nullptr;
    // preconditionner service
    std::string precond_type_s = vm["precond"].as<std::string>();
    if (precond_type_s.compare("bjacobi") == 0) {
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigJacobi>();
      prec = std::make_shared<Alien::PETScPrecConfigJacobiService>(pm, options_prec);
    } else if (precond_type_s.compare("diag") == 0) {
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigDiagonal>();
      prec = std::make_shared<Alien::PETScPrecConfigDiagonalService>(pm, options_prec);
    } else if (precond_type_s.compare("spai") == 0) {
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigSPAI>();
      prec = std::make_shared<Alien::PETScPrecConfigSPAIService>(pm, options_prec);
    } else if (precond_type_s.compare("none") == 0) {
      auto options_prec =
          std::make_shared<StrongOptionsPETScPrecConfigNoPreconditioner>();
      prec = std::make_shared<Alien::PETScPrecConfigNoPreconditionerService>(
          pm, options_prec);
    }
    std::string solver = vm["solver"].as<std::string>();
    if (solver.compare("bicgs") == 0) {
      // solver service bicgs
      using namespace PETScSolverConfigBiCGStabOptionsNames;
      auto options_solver = std::make_shared<StrongOptionsPETScSolverConfigBiCGStab>(
          _numIterationsMax = max_iter, _stopCriteriaValue = tol, _preconditioner = prec);
      // root petsc option
      auto root_options = std::make_shared<StrongOptionsPETScLinearSolver>(
          PETScLinearSolverOptionsNames::_solver =
              std::make_shared<Alien::PETScSolverConfigBiCGStabService>(
                  pm, options_solver));
      // root petsc service
      return std::make_shared<Alien::PETScLinearSolverService>(pm, root_options);
    }
    if (solver.compare("lu") == 0) {
      // solver service lu
      using namespace PETScSolverConfigLUOptionsNames;
      auto options_solver = std::make_shared<StrongOptionsPETScSolverConfigLU>();
      // root petsc option
      auto root_options = std::make_shared<StrongOptionsPETScLinearSolver>(
          PETScLinearSolverOptionsNames::_solver =
              std::make_shared<Alien::PETScSolverConfigLUService>(pm, options_solver));
      // root petsc service
      return std::make_shared<Alien::PETScLinearSolverService>(pm, root_options);
    }
    if (solver.compare("mumps") == 0) {
#ifdef PETSC_HAVE_MUMPS
        // solver service mumps
        auto options_solver = std::make_shared<IOptionsPETScSolverConfigMUMPS>();
        // root petsc option
        auto root_options = std::make_shared<StrongOptionsPETScLinearSolver>(
                PETScLinearSolverOptionsNames::_solver =
                        std::make_shared<Alien::PETScSolverConfigMUMPSService>(pm, options_solver));
        // root petsc service
        return std::make_shared<Alien::PETScLinearSolverService>(pm, root_options);
#endif
    }
    tm->fatal() << "*** solver " << solver << " not available in test!";
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }

  if (solver_package.compare("hypre") == 0) {
#ifdef ALIEN_USE_HYPRE
    std::string solver_type_s = vm["solver"].as<std::string>();
    HypreOptionTypes::eSolver solver_type =
        OptionsHypreSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = vm["precond"].as<std::string>();
    HypreOptionTypes::ePreconditioner precond_type =
        OptionsHypreSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    using namespace HypreSolverOptionsNames;
    auto options = std::make_shared<StrongOptionsHypreSolver>(
        _numIterationsMax = max_iter, _stopCriteriaValue = tol, _solver = solver_type,
        _preconditioner = precond_type);
    // service
    return std::make_shared<Alien::HypreLinearSolver>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }

  if (solver_package.compare("ifpsolver") == 0) {
#ifdef ALIEN_USE_IFPSOLVER
    std::string precond_type_s = vm["precond"].as<std::string>();
    IFPSolverProperty::ePrecondType precond_type =
        OptionsIFPLinearSolverUtils::stringToPrecondOptionEnum(precond_type_s);
    // options
    auto options = std::make_shared<StrongOptionsIFPLinearSolver>(
        IFPLinearSolverOptionsNames::_output = vm["output-level"].as<int>(),
        IFPLinearSolverOptionsNames::_numIterationsMax = max_iter,
        IFPLinearSolverOptionsNames::_stopCriteriaValue = tol,
        IFPLinearSolverOptionsNames::_precondOption = precond_type);
    // service
    return std::make_shared<Alien::IFPLinearSolverService>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }

  if (solver_package.compare("mcgsolver") == 0) {
#ifdef ALIEN_USE_MCGSOLVER
    std::string precond_type_s = vm["precond"].as<std::string>();
    MCGSolver::ePrecondType precond_type =
        OptionsMCGSolverUtils::stringToPreconditionerEnum(precond_type_s);    std::string kernel_type_s = vm["kernel"].as<std::string>();
    MCGOptionTypes::eKernelType kernel_type =
        OptionsMCGSolverUtils::stringToKernelEnum(kernel_type_s);
    // options
    auto options = std::make_shared<StrongOptionsMCGSolver>(
        MCGSolverOptionsNames::_output = vm["output-level"].as<int>(),
        MCGSolverOptionsNames::_maxIterationNum = max_iter,
        MCGSolverOptionsNames::_stopCriteriaValue = tol,
        MCGSolverOptionsNames::_kernel = kernel_type,
        MCGSolverOptionsNames::_preconditioner = precond_type);
    // service
    return std::make_shared<Alien::MCGLinearSolver>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }

  if (solver_package.compare("hts") == 0) {
#ifdef ALIEN_USE_HTSSOLVER
    std::string solver_type_s = vm["solver"].as<std::string>();
    HTSOptionTypes::eSolver solver_type =
        OptionsHTSSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = vm["precond"].as<std::string>();
    HTSOptionTypes::ePreconditioner precond_type =
        OptionsHTSSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    using namespace HTSSolverOptionsNames;
    auto options = std::make_shared<StrongOptionsHTSSolver>(
        _maxIterationNum = max_iter, _stopCriteriaValue = tol, _solver = solver_type,
        _preconditioner = precond_type);
    // service
    return std::make_shared<Alien::HTSLinearSolver>(pm, options);
#else
    tm->fatal() << "*** package " << solver_package << " not available!";
#endif
  }

  if (solver_package.compare("mtlsolver") == 0) {
#ifdef ALIEN_USE_MTL4
    std::string solver_type_s = vm["solver"].as<std::string>();
    MTLOptionTypes::eSolver solver_type =
        OptionsMTLLinearSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = vm["precond"].as<std::string>();
    MTLOptionTypes::ePreconditioner precond_type =
        OptionsMTLLinearSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    auto options = std::make_shared<StrongOptionsMTLLinearSolver>(
        MTLLinearSolverOptionsNames::_outputLevel = vm["output-level"].as<int>(),
        MTLLinearSolverOptionsNames::_maxIterationNum = max_iter,
        MTLLinearSolverOptionsNames::_stopCriteriaValue = tol,
        MTLLinearSolverOptionsNames::_preconditioner = precond_type,
        MTLLinearSolverOptionsNames::_solver = solver_type);
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
