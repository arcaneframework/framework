/*
 * SolverFabric.h
 *
 *  Created on: Nov 29, 2020
 *      Author: gratienj
 */

#ifndef MODULES_INTERFACE_C_SRC_ALIEN_C_SOLVERFABRIC_H_
#define MODULES_INTERFACE_C_SRC_ALIEN_C_SOLVERFABRIC_H_


#ifdef ALIEN_USE_MTL4
#include <alien/kernels/mtl/linear_solver/arcane/MTLLinearSolverService.h>
#include <alien/kernels/mtl/linear_solver/MTLOptionTypes.h>
#include <ALIEN/axl/MTLLinearSolver_IOptions.h>
#include <ALIEN/axl/MTLLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScLinearSolverService.h>
#include <alien/kernels/petsc/linear_solver/PETScInternalLinearSolver.h>
// preconditionner
#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigDiagonalService.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigJacobiService.h>
#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigNoPreconditionerService.h>
#include <ALIEN/axl/PETScPrecConfigDiagonal_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigDiagonal_StrongOptions.h>
#include <ALIEN/axl/PETScPrecConfigJacobi_IOptions.h>
#include <ALIEN/axl/PETScPrecConfigJacobi_StrongOptions.h>
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
// root linear solver instance
#include <ALIEN/axl/PETScLinearSolver_IOptions.h>
#include <ALIEN/axl/PETScLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_HYPRE
#include <alien/kernels/hypre/linear_solver/HypreOptionTypes.h>
#include <alien/kernels/hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/axl/HypreSolver_IOptions.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_IFPSOLVER
#include <alien/kernels/ifp/linear_solver/arcane/IFPLinearSolverService.h>
#include <alien/kernels/ifp/linear_solver/IFPSolverProperty.h>
#include <ALIEN/axl/IFPLinearSolver_IOptions.h>
#include <ALIEN/axl/IFPLinearSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_MCGSOLVER
#include <alien/kernels/mcg/linear_solver/arcane/MCGLinearSolver.h>
#include <alien/kernels/mcg/linear_solver/MCGOptionTypes.h>
#include <ALIEN/axl/MCGSolver_IOptions.h>
#include <ALIEN/axl/MCGSolver_StrongOptions.h>
#endif


#include <boost/program_options/variables_map.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

class BaseSolverOptions
{
public :
  template<typename T>
  static T get(boost::program_options::variables_map const& options, std::string const& key)
  {
    return options[key].as<T>() ;
  }

  template<typename T>
  static T get(boost::property_tree::ptree const& options, std::string const& key)
  {
    return options.get<T>(key) ;
  }

};

template<typename tag>
class SolverFabric ;


#ifdef ALIEN_USE_PETSC
template<>
class SolverFabric<Alien::BackEnd::tag::petsc>
: public BaseSolverOptions
{
public :

  static void
  add_options(boost::program_options::options_description& cmdline_options)
  {
    using namespace boost::program_options;
    options_description desc("PETSC options : ");
    desc.add_options()("petsc-solver", value<std::string>()->default_value("bicgs"),"solver algo name : bicgs lu")
        ("petsc-precond", value<std::string>()->default_value("none"),"preconditioner bjacobi diag none");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  static Alien::ILinearSolver* create(OptionT const& options,Alien::IMessagePassingMng* pm)
  {
    double tol = get<double>(options,"tol");
    int max_iter = get<int>(options,"max-iter");

    std::shared_ptr<Alien::IPETScPC> prec = nullptr;
    // preconditionner service
    std::string precond_type_s = get<std::string>(options,"petsc-precond");
    if (precond_type_s.compare("bjacobi") == 0)
    {
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigJacobi>();
      prec = std::make_shared<Alien::PETScPrecConfigJacobiService>(pm, options_prec);
    }
    else if (precond_type_s.compare("diag") == 0)
    {
      auto options_prec = std::make_shared<StrongOptionsPETScPrecConfigDiagonal>();
      prec = std::make_shared<Alien::PETScPrecConfigDiagonalService>(pm, options_prec);
    }
    else if (precond_type_s.compare("none") == 0)
    {
      auto options_prec =
          std::make_shared<StrongOptionsPETScPrecConfigNoPreconditioner>();
      prec = std::make_shared<Alien::PETScPrecConfigNoPreconditionerService>(
          pm, options_prec);
    }
    std::string solver = get<std::string>(options,"petsc-solver");
    if (solver.compare("bicgs") == 0)
    {
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
      return new Alien::PETScLinearSolverService(pm, root_options);
    }
    if (solver.compare("lu") == 0)
    {
      // solver service lu
      using namespace PETScSolverConfigLUOptionsNames;
      auto options_solver = std::make_shared<StrongOptionsPETScSolverConfigLU>();
      // root petsc option
      auto root_options = std::make_shared<StrongOptionsPETScLinearSolver>(
          PETScLinearSolverOptionsNames::_solver =
              std::make_shared<Alien::PETScSolverConfigLUService>(pm, options_solver));
      // root petsc service
      return new Alien::PETScLinearSolverService(pm, root_options);
    }
    return nullptr ;
  }
};
#endif


#ifdef ALIEN_USE_HYPRE
template<>
class SolverFabric<Alien::BackEnd::tag::hypre>
: public BaseSolverOptions
{
public :

  static void
  add_options(boost::program_options::options_description& cmdline_options)
  {
    using namespace boost::program_options;
    options_description desc("HYPRE options : ");
    desc.add_options()("hypre-solver", value<std::string>()->default_value("bicgs"),"solver algo name : amg cg gmres bicgstab")
        ("hypre-precond", value<std::string>()->default_value("none"),"preconditioner none diag amg parasails euclid");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  static Alien::ILinearSolver* create(OptionT const& options,Alien::IMessagePassingMng* pm)
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
};
#endif


#ifdef ALIEN_USE_IFPSOLVER
template<>
class SolverFabric<Alien::BackEnd::tag::ifpsolver>
: public BaseSolverOptions
{
public :
  static void
  add_options(boost::program_options::options_description& cmdline_options)
  {
    using namespace boost::program_options;
    options_description desc("IFPSolver options : ");
    desc.add_options()("ifps-solver", value<std::string>()->default_value("bicgs"),"solver algo name : bicgs lu")
        ("ifps-precond", value<std::string>()->default_value("none"),"preconditioner diag none ilu0 amg cpramg");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  static Alien::ILinearSolver* create(OptionT const& options,Alien::IMessagePassingMng* pm)
  {
    double tol = get<double>(options,"tol");
    int max_iter = get<int>(options,"max-iter");

    std::string precond_type_s = get<std::string>(options,"ifps-precond");
    IFPSolverProperty::ePrecondType precond_type =
        OptionsIFPLinearSolverUtils::stringToPrecondOptionEnum(precond_type_s);
    // options
    auto solver_options = std::make_shared<StrongOptionsIFPLinearSolver>(
        IFPLinearSolverOptionsNames::_output = get<int>(options,"output-level"),
        IFPLinearSolverOptionsNames::_numIterationsMax = max_iter,
        IFPLinearSolverOptionsNames::_stopCriteriaValue = tol,
        IFPLinearSolverOptionsNames::_precondOption = precond_type);
    // service
    return  new Alien::IFPLinearSolverService(pm, solver_options);

  }
};
#endif

#ifdef ALIEN_USE_MTL4
template<>
class SolverFabric<Alien::BackEnd::tag::mtl>
: public BaseSolverOptions
{
public :

  static void
  add_options(boost::program_options::options_description& cmdline_options)
  {
    using namespace boost::program_options;
    options_description desc("MTL4 options : ");
    desc.add_options()("mtl4-solver", value<std::string>()->default_value("bicgs"),"solver algo name : bicgstab")
        ("mtl4-precond", value<std::string>()->default_value("none"),"preconditioner ilu diag none");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  static Alien::ILinearSolver* create(OptionT const& options,Alien::IMessagePassingMng* pm)
  {
    double tol = get<double>(options,"tol");
    int max_iter = get<int>(options,"max-iter");

    std::string solver_type_s = get<std::string>(options,"mtl4-solver");
    MTLOptionTypes::eSolver solver_type =
        OptionsMTLLinearSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = get<std::string>(options,"mtl4-precond");
    MTLOptionTypes::ePreconditioner precond_type =
        OptionsMTLLinearSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    auto solver_options = std::make_shared<StrongOptionsMTLLinearSolver>(
        MTLLinearSolverOptionsNames::_outputLevel = get<int>(options,"output-level"),
        MTLLinearSolverOptionsNames::_maxIterationNum = max_iter,
        MTLLinearSolverOptionsNames::_stopCriteriaValue = tol,
        MTLLinearSolverOptionsNames::_preconditioner = precond_type,
        MTLLinearSolverOptionsNames::_solver = solver_type);
    // service
    return new Alien::MTLLinearSolverService(pm, solver_options);

  }
};
#endif


#ifdef ALIEN_USE_MCGSOLVER
template<>
class SolverFabric<Alien::BackEnd::tag::mcgsolver>
: public BaseSolverOptions
{
public :
  static void
  add_options(boost::program_options::options_description& cmdline_options)
  {
    using namespace boost::program_options;
    options_description desc("MCGSolver options : ");
    desc.add_options()
        ("mcgs-solver",  value<std::string>()->default_value("bicgs"),"solver algo name : bicgs lu")
        ("mcgs-precond", value<std::string>()->default_value("none"),"preconditioner bjacobi diag none")
        ("kernel",       value<std::string>()->default_value("mcgkernel"),"mcgsolver kernel name");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  static Alien::ILinearSolver* create(OptionT const& options,Alien::IMessagePassingMng* pm)
  {
    double tol = get<double>(options,"tol");
    int max_iter = get<int>(options,"max-iter");
    std::string precond_type_s = get<std::string>("mcgs-precond");
    MCGOptionTypes::ePreconditioner precond_type =
        OptionsMCGSolverUtils::stringToPreconditionerEnum(precond_type_s);
    std::string kernel_type_s = vm["kernel"].as<std::string>();
    MCGOptionTypes::eKernelType kernel_type =
        OptionsMCGSolverUtils::stringToKernelEnum(kernel_type_s);
    // options
    auto solver_options = std::make_shared<StrongOptionsMCGSolver>(
        MCGSolverOptionsNames::_output = get<int>("output-level"),
        MCGSolverOptionsNames::_maxIterationNum = max_iter,
        MCGSolverOptionsNames::_stopCriteriaValue = tol,
        MCGSolverOptionsNames::_kernel = kernel_type,
        MCGSolverOptionsNames::_preconditioner = precond_type);
    // service
    return new Alien::MCGLinearSolver(pm, solver_options);
  }
} ;
#endif

#endif /* MODULES_INTERFACE_C_SRC_ALIEN_C_SOLVERFABRIC_H_ */
