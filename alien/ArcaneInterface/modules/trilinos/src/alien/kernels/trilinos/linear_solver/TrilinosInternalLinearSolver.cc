﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>


#include <alien/data/Space.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/ref/AlienRefSemantic.h>

#include "alien/kernels/trilinos/TrilinosPrecomp.h"
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>
#include <alien/kernels/trilinos/algebra/TrilinosLinearAlgebra.h>

#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>

#include <alien/kernels/trilinos/linear_solver/TrilinosOptionTypes.h>
#include <alien/AlienTrilinosPrecomp.h>

#include <ALIEN/axl/TrilinosSolver_IOptions.h>
#include <alien/kernels/trilinos/linear_solver/TrilinosInternalSolver.h>
#include <alien/kernels/trilinos/linear_solver/TrilinosInternalLinearSolver.h>
#include <alien/core/backend/LinearSolverT.h>
#include <alien/core/backend/SolverFabricRegisterer.h>
#include <alien/core/block/ComputeBlockOffsets.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
const std::string TrilinosOptionTypes::solver_type[NumOfSolver] = { "BICGSTAB", "CG",
  "GMRES", "ML", "MueLU", "KLU2" };

const std::string TrilinosOptionTypes::preconditioner_type[NumOfPrecond] = {
  "None", "RELAXATION", "CHEBYSHEV", "RILUK", "ILUT", "FAST_ILU", "SCHWARZ", "ML",
  "MueLu",
};

namespace Alien {

#ifdef KOKKOS_ENABLE_SERIAL
template class ALIEN_TRILINOS_EXPORT LinearSolver<BackEnd::tag::tpetraserial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class ALIEN_TRILINOS_EXPORT LinearSolver<BackEnd::tag::tpetraomp>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class ALIEN_TRILINOS_EXPORT LinearSolver<BackEnd::tag::tpetrapth>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class ALIEN_TRILINOS_EXPORT LinearSolver<BackEnd::tag::tpetracuda>;
#endif

/*---------------------------------------------------------------------------*/
template <typename TagT>
TrilinosInternalLinearSolver<TagT>::TrilinosInternalLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    IOptionsTrilinosSolver* options)
: m_parallel_mng(parallel_mng)
, m_options(options)
{
}

template <typename TagT>
void
TrilinosInternalLinearSolver<TagT>::init([[maybe_unused]] int argc,[[maybe_unused]]  char const** argv)
{
}

template <typename TagT>
String
TrilinosInternalLinearSolver<TagT>::getBackEndName() const
{
  return AlgebraTraits<TagT>::name();
}

/*---------------------------------------------------------------------------*/
template <typename TagT>
void
TrilinosInternalLinearSolver<TagT>::init()
{
  if(m_initialized) return ;
  bool use_amgx = false;
  if (m_options->muelu().size() > 0 && m_options->muelu()[0]->amgx().size() > 0
      && m_options->muelu()[0]->amgx()[0]->enable())
    use_amgx = true;
  TrilinosInternal::TrilinosInternal::initialize(m_parallel_mng,
      TrilinosInternal::TrilinosInternal::Node<TagT>::execution_space_name,
      m_options->nbThreads(), use_amgx);
  assert(TrilinosInternal::TrilinosInternal::Node<TagT>::execution_space_name.compare(
             TrilinosInternal::TrilinosInternal::getExecutionSpace())
      == 0);

  traceMng()->info() << "TRILINOS EXECUTION SPACE : "
                     << TrilinosInternal::TrilinosInternal::getExecutionSpace();
  traceMng()->info() << "TRILINOS NB THREADS      : "
                     << TrilinosInternal::TrilinosInternal::getNbThreads();
  m_output_level = m_options->output();

  m_trilinos_solver.reset(new TrilinosInternal::SolverInternal<TagT>());
  
#ifdef TEST
  if(use_amgx)
  {
      /* create config */
      auto& env = m_trilinos_solver->m_amgx_env ;
      std::string amgx_config_file("PBICGSTAB_CLASSICAL_JACOBI.json") ;
      AMGX_SAFE_CALL(AMGX_config_create_from_file(&env.m_config, amgx_config_file.c_str()));

      std::cout<<"CREATE AMGX RESOURCES"<<std::endl ;
      AMGX_resources_create_simple(&env.m_resources,env.m_config);
      env.m_mode = AMGX_mode_dDDI;
      AMGX_solver_create(&env.m_solver, env.m_resources, env.m_mode,env.m_config);
      AMGX_matrix_create(&env.m_A,      env.m_resources, env.m_mode);
      AMGX_vector_create(&env.m_X,      env.m_resources, env.m_mode);
      AMGX_vector_create(&env.m_Y,      env.m_resources, env.m_mode);
      std::cout<<"END CREATE AMGX RESOURCES"<<std::endl ;
  }
#endif
  m_precond_name = TrilinosOptionTypes::precondName(m_options->preconditioner());
  auto* mpi_mng =
      dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(m_parallel_mng);
  const MPI_Comm* comm = static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());
  m_trilinos_solver->initPrecondParameters(m_options, comm);

  m_solver_name = TrilinosOptionTypes::solverName(m_options->solver());
  m_trilinos_solver->initSolverParameters(m_options);
  m_initialized = true ;
}

template <typename TagT>
void
TrilinosInternalLinearSolver<TagT>::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  m_parallel_mng = pm;
}

template <typename TagT>
bool
TrilinosInternalLinearSolver<TagT>::solve(
    const CSRMatrixType& matrix, const CSRVectorType& b, CSRVectorType& x)
{
  typename TrilinosInternal::SolverInternal<TagT>::Status status =
      m_trilinos_solver->solve(*matrix.internal()->m_internal,
                                matrix.internal()->m_coordinates,
                               *b.internal()->m_internal,
                               *x.internal()->m_internal);
  m_status.succeeded = status.m_converged;
  m_status.iteration_count = status.m_num_iters;
  m_status.residual = status.m_residual;
  return m_status.succeeded;
}
/*---------------------------------------------------------------------------*/

template <typename TagT>
void
TrilinosInternalLinearSolver<TagT>::end()
{
  if (m_output_level > 0) {
    internalPrintInfo();
  }
}

template <typename TagT>
const Alien::SolverStatus&
TrilinosInternalLinearSolver<TagT>::getStatus() const
{
  if (m_output_level > 0) {
    printInfo();
  }
  return m_status;
}

template <typename TagT>
void
TrilinosInternalLinearSolver<TagT>::internalPrintInfo() const
{
  m_stat.print(const_cast<ITraceMng*>(traceMng()), m_status,
      Arccore::String::format("Linear Solver : {0}", "TrilinosSolver"));
}

template <typename TagT>
std::shared_ptr<ILinearAlgebra>
TrilinosInternalLinearSolver<TagT>::algebra() const
{
  //return std::shared_ptr<ILinearAlgebra>(new Alien::TrilinosLinearAlgebra());
  return std::shared_ptr<ILinearAlgebra>(new Alien::LinearAlgebra<TagT>());
}

#ifdef KOKKOS_ENABLE_SERIAL
template class TrilinosInternalLinearSolver<BackEnd::tag::tpetraserial>;

IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetraserial>,
    TrilinosVector<Real, BackEnd::tag::tpetraserial>>*
TrilinosInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsTrilinosSolver* options)
{
  return new TrilinosInternalLinearSolver<BackEnd::tag::tpetraserial>(p_mng, options);
}
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosInternalLinearSolver<BackEnd::tag::tpetraomp>;

IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetraomp>,
    TrilinosVector<Real, BackEnd::tag::tpetraomp>>*
TpetraOmpInternalLinearSolverFactory(Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsTrilinosSolver* options)
{
  return new TrilinosInternalLinearSolver<BackEnd::tag::tpetraomp>(p_mng, options);
}
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosInternalLinearSolver<BackEnd::tag::tpetrapth>;

IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetrapth>,
    TrilinosVector<Real, BackEnd::tag::tpetrapth>>*
TpetraPthInternalLinearSolverFactory(Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsTrilinosSolver* options)
{
  return new TrilinosInternalLinearSolver<BackEnd::tag::tpetrapth>(p_mng, options);
}
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class TrilinosInternalLinearSolver<BackEnd::tag::tpetracuda>;

IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetracuda>,
    TrilinosVector<Real, BackEnd::tag::tpetracuda>>*
TpetraCudaInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsTrilinosSolver* options)
{
  return new TrilinosInternalLinearSolver<BackEnd::tag::tpetracuda>(p_mng, options);
}
#endif



}

#include <alien/kernels/trilinos/linear_solver/TrilinosInternalLinearSolver.h>
#include <alien/kernels/trilinos/linear_solver/TrilinosOptionTypes.h>
#include <alien/kernels/trilinos/linear_solver/arcane/TrilinosLinearSolver.h>
#include <ALIEN/axl/TrilinosSolver_axl.h>
#include <ALIEN/axl/TrilinosSolver_StrongOptions.h>
#include <ALIEN/axl/TrilinosSolver_IOptions.h>

namespace Alien {

template<>
#ifdef KOKKOS_ENABLE_SERIAL
class SolverFabric<Alien::BackEnd::tag::tpetraserial>
#else
#ifdef KOKKOS_ENABLE_OPENMP
class SolverFabric<Alien::BackEnd::tag::tpetraomp>
#endif
#endif
: public ISolverFabric
{
public :
  BackEndId backend() const {
     return "trilinos" ;
  }

  void
  add_options(CmdLineOptionDescType& cmdline_options) const
  {
    using namespace boost::program_options;
    options_description desc("TrilinosSolver options");
    desc.add_options()
        ("tri-solver",                   value<std::string>()->default_value("bicgs"),"solver algo name : bicgstab cg gmres ml muelu klu2 ")
        ("tri-precond",                  value<std::string>()->default_value("iluk"),"precond algo name : none relaxation chebyshev iluk ilut filu schwarz ml muelu")
        ("tri-iluk-level-of-fill",       value<int>()->default_value(0),"iluk level of fill")
        ("tri-iluk-relax-value",         value<double>()->default_value(0),"iluk relax value")
        ("tri-iluk-absolute-threshold",  value<double>()->default_value(0.),"iluk absolute threshold")
        ("tri-iluk-relative-threshold",  value<double>()->default_value(0.),"iluk relative threshold")
        ("tri-ilut-level-of-fill",       value<int>()->default_value(0),"ilut level of fill")
        ("tri-ilut-drop-tolerance",      value<double>()->default_value(0.),"ilut drop tolerance")
        ("tri-ilut-absolute-threshold",  value<double>()->default_value(0.),"ilut absolute threshold")
        ("tri-ilut-relative-threshold",  value<double>()->default_value(1.),"ilut relative threshold")
        ("tri-ilut-relax-value",         value<double>()->default_value(0.),"ilut relax value")
        ("tri-filu-level",               value<int>()->default_value(0),    "filu level of fill")
        ("tri-filu-damping-factor",      value<double>()->default_value(0.),"filu damping factor")
        ("tri-filu-solver-nb-iterations",value<int>()->default_value(1),    "filu solver nb iterations")
        ("tri-filu-factor-nb-iterations",value<int>()->default_value(1),    "filu factor nb iterations")
        ("tri-relax-type",               value<std::string>()->default_value("Jacobi"), "relaxation type Jacobi, Gauss-Seidel, Symmetric  Gauss-Seidel")
        ("tri-relax-sweeps",             value<int>()->default_value(0),     "relaxation sweeps")
        ("tri-relax-factor",             value<double>()->default_value(1.),"relaxation factor")
        ("tri-relax-backward-mode",      value<int>()->default_value(0),    "relaxation backward mode")
        ("tri-relax-use-l1",             value<int>()->default_value(0),    "relaxaxation use l1")
        ("tri-relax-l1-eta",             value<double>()->default_value(1.5),"relaxation l1 factor eta")
        ("tri-relax-zero-starting-solution",value<int>()->default_value(0), "relaxation null init solution")
        ("tri-chebyshev-degree",            value<int>()->default_value(1),  "chebyshev polynome degree")
        ("tri-chebyshev-max-eigenvalue",    value<double>()->default_value(0.),"chebyshev max eigen value")
        ("tri-chebyshev-ratio-eigenvalue",  value<double>()->default_value(30.),"chebychev eigenvalue ratio")
        ("tri-chebyshev-eigenvalue-max-iteration",value<int>()->default_value(10),"chebyshev max ieration num")
        ("tri-chebyshev-boost-factor",      value<double>()->default_value(1.1),"chebyshev-boost-factor")
        ("tri-chebyshev-zero-starting-solution",value<int>()->default_value(1),"chebyshev-zero-starting-solution")
        ("tri-schwarz-subdomain-solver",   value<std::string>()->default_value("undefined"),"schwarz-subdomain-solver")
        ("tri-schwarz-combine-mode",       value<std::string>()->default_value("ZERO"),"schwarz-combine-mode")
        ("tri-schwarz-num-iterations",     value<int>()->default_value(1),     "schwarz-num-iterations")
        ("tri-muelu-max-level",            value<int>()->default_value(10),     "muelu-max-level")
        ("tri-muelu-max-coarse-size",      value<int>()->default_value(10),     "muelu-max-coarse-size")
        ("tri-muelu-cycle-type",           value<std::string>()->default_value("V"),"muelu-cycle-type")
        ("tri-muelu-symmetric",            value<int>()->default_value(1),     "muelu-symmetric")
        ("tri-muelu-xml-parameter-file",   value<std::string>()->default_value("undefined"),"muelu-xml-parameter-file")
        ("tri-muelu-smoother-type",        value<std::string>()->default_value("RELAXATION"),"muelu-smoother-type")
        ("tri-muelu-smoother-overlap",     value<int>()->default_value(0),     "muelu-smoother-overlap")
        ("tri-muelu-coarse-type",          value<std::string>()->default_value("SuperLU"),"muelu-coarse-overlap")
        ("tri-muelu-coarse-overlap",       value<int>()->default_value(0),     "muelu-coarse-overlap")
        ("tri-muelu-aggegation-type",      value<std::string>()->default_value("uncoupled"),"muelu-aggegation-type")
        ("tri-muelu-aggregation-mode",     value<std::string>()->default_value("uncoupled"),"muelu-aggregation-ordering")
        ("tri-muelu-aggregation-ordering", value<std::string>()->default_value("natural"),"muelu-aggregation-drop-scheme")
        ("tri-muelu-aggregation-drop-scheme",value<std::string>()->default_value("classical"),"muelu-aggregation-drop-scheme")
        ("tri-muelu-aggregation-drop-tol", value<double>()->default_value(0.),"muelu-aggregation-drop-tol")
        ("tri-muelu-aggregation-min-agg-size",value<int>()->default_value(2),"muelu-aggregation-min-agg-size")
        ("tri-muelu-aggregation-max-agg-size",value<int>()->default_value(-1),"muelu-aggregation-max-agg-size")
        ("tri-muelu-aggregation-dirichlet-threshold",value<double>()->default_value(0.),"muelu-aggregation-dirichlet-threshold")
        ("tri-muelu-multigrid-algorithm",  value<std::string>()->default_value("sa"),"muelu-multigrid-algorithm")
        ("tri-muelu-sa-dampling-factor",   value<double>()->default_value(1.33),"muelu-sa-dampling-factor") ;

   cmdline_options.add(desc) ;
  }


  template<typename OptionT>
  Alien::ILinearSolver* _create(OptionT const& options,Alien::IMessagePassingMng* pm) const
  {
    int output_level = get<int>(options,   "output-level") ;
    double tol       = get<double>(options,"tol");
    int max_iter     = get<int>(options,   "max-iter");

    std::string precond_type_s = get<std::string>(options,"trilinos-precond");
    TrilinosOptionTypes::ePreconditioner precond_type =
        OptionsTrilinosSolverUtils::stringToPreconditionerEnum(precond_type_s);

    std::string solver_type_s = get<std::string>(options,"trilinos-solver");
    TrilinosOptionTypes::eSolver solver_type =
        OptionsTrilinosSolverUtils::stringToSolverEnum(solver_type_s);

    auto options_iluk = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsILUKOptType>(
        TrilinosSolverOptionsNames::ILUKOptTypeOptionsNames::_levelOfFill       = get<int>(options,"tri-iluk-level-of-fill"),
        TrilinosSolverOptionsNames::ILUKOptTypeOptionsNames::_relaxValue        = get<double>(options,"tri-iluk-relax-value"),
        TrilinosSolverOptionsNames::ILUKOptTypeOptionsNames::_absoluteThreshold = get<double>(options,"tri-iluk-absolute-threshold"),
        TrilinosSolverOptionsNames::ILUKOptTypeOptionsNames::_relativeThreshold = get<double>(options,"tri-iluk-relative-threshold")
        ) ;

    auto options_ilut = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsILUTOptType>(
            TrilinosSolverOptionsNames::ILUTOptTypeOptionsNames::_levelOfFill       = get<int>(options,"tri-ilut-level-of-fill"),
            TrilinosSolverOptionsNames::ILUTOptTypeOptionsNames::_dropTolerance     = get<double>(options,"tri-ilut-drop-tolerance"),
            TrilinosSolverOptionsNames::ILUTOptTypeOptionsNames::_absoluteThreshold = get<double>(options,"tri-ilut-absolute-threshold"),
            TrilinosSolverOptionsNames::ILUTOptTypeOptionsNames::_relativeThreshold = get<double>(options,"tri-ilut-relative-threshold"),
            TrilinosSolverOptionsNames::ILUTOptTypeOptionsNames::_relaxValue        = get<double>(options,"tri-ilut-relax-value")
    ) ;

    auto options_filu = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsFILUOptType>(
            TrilinosSolverOptionsNames::FILUOptTypeOptionsNames::_level              = get<int>(options,"tri-filu-level"),
            TrilinosSolverOptionsNames::FILUOptTypeOptionsNames::_dampingFactor      = get<double>(options,"tri-filu-damping-factor"),
            TrilinosSolverOptionsNames::FILUOptTypeOptionsNames::_solverNbIterations = get<double>(options,"tri-filu-solver-nb-iterations"),
            TrilinosSolverOptionsNames::FILUOptTypeOptionsNames::_factorNbIterations = get<double>(options,"tri-filu-factor-nb-iterations")
            ) ;

    auto options_relaxation = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsRelaxationOptType>(
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_type          = get<std::string>(options,"tri-relax-type"),
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_sweeps        = get<int>(options,"tri-relax-sweeps"),
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_dampingFactor = get<double>(options,"tri-relax-factor"),
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_backwardMode  = (get<int>(options,"tri-relax-backward-mode") == 1),
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_useL1         = (get<int>(options,"tri-relax-use-l1") == 1),
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_l1Eta         = get<double>(options,"tri-relax-l1-eta"),
            TrilinosSolverOptionsNames::RelaxationOptTypeOptionsNames::_zeroStartingSolution = (get<int>(options,"tri-relax-zero-starting-solution") == 1)
            ) ;

    auto options_chebyshev = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsChebyshevOptType>(
            TrilinosSolverOptionsNames::ChebyshevOptTypeOptionsNames::_degree          = get<int>(options,"tri-chebyshev-degree"),
            TrilinosSolverOptionsNames::ChebyshevOptTypeOptionsNames::_maxEigenvalue   = get<double>(options,"tri-chebyshev-max-eigenvalue"),
            TrilinosSolverOptionsNames::ChebyshevOptTypeOptionsNames::_ratioEigenvalue = get<double>(options,"tri-chebyshev-ratio-eigenvalue"),
            TrilinosSolverOptionsNames::ChebyshevOptTypeOptionsNames::_eigenvalueMaxIterations  = get<int>(options,"tri-chebyshev-eigenvalue-max-iteration"),
            TrilinosSolverOptionsNames::ChebyshevOptTypeOptionsNames::_boostFactor              = get<double>(options,"tri-chebyshev-boost-factor"),
            TrilinosSolverOptionsNames::ChebyshevOptTypeOptionsNames::_zeroStartingSolution     = (get<int>(options,"tri-relax-zero-starting-solution") == 1)
            ) ;


    auto options_schwarz = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsSchwarzOptType>(
            TrilinosSolverOptionsNames::SchwarzOptTypeOptionsNames::_subdomainSolver  = get<std::string>(options,"tri-schwarz-subdomain-solver"),
            TrilinosSolverOptionsNames::SchwarzOptTypeOptionsNames::_combineMode      = get<std::string>(options,"tri-schwarz-combine-mode"),
            TrilinosSolverOptionsNames::SchwarzOptTypeOptionsNames::_numIterations    = get<int>(options,"tri-schwarz-num-iterations")
            ) ;

    auto options_muelu = std::make_shared<TrilinosSolverOptionsNames::StrongOptionsMueLUOptType>(
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_maxLevel           = get<int>(options,"tri-muelu-max-level"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_maxCoarseSize      = get<int>(options,"tri-muelu-max-coarse-size"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_cycleType          = get<std::string>(options,"tri-muelu-cycle-type"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_symmetric          = get<int>(options,"tri-muelu-symmetric"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_xmlParameterFile   = get<std::string>(options,"tri-muelu-xml-parameter-file"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_smootherType       = get<std::string>(options,"tri-muelu-smoother-type"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_smootherOverlap    = get<int>(options,"tri-muelu-smoother-overlap"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_coarseType         = get<std::string>(options,"tri-muelu-coarse-type"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_coarseOverlap      = get<int>(options,"tri-muelu-coarse-overlap"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationType    = get<std::string>(options,"tri-muelu-aggegation-type"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationMode    = get<std::string>(options,"tri-muelu-aggregation-mode"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationOrdering = get<std::string>(options,"tri-muelu-aggregation-ordering"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationDropScheme = get<std::string>(options,"tri-muelu-aggregation-drop-scheme"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationDropTol  = get<double>(options,"tri-muelu-aggregation-drop-tol"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationMinAggSize = get<int>(options,"tri-muelu-aggregation-min-agg-size"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationMaxAggSize = get<int>(options,"tri-muelu-aggregation-max-agg-size"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_aggregationDirichletThreshold = get<double>(options,"tri-muelu-aggregation-dirichlet-threshold"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_multigridAlgorithm     = get<std::string>(options,"tri-muelu-multigrid-algorithm"),
            TrilinosSolverOptionsNames::MueLUOptTypeOptionsNames::_saDamplingFactor       = get<double>(options,"tri-muelu-sa-dampling-factor")
    ) ;

    // options
    auto solver_options = std::make_shared<StrongOptionsTrilinosSolver>(
        TrilinosSolverOptionsNames::_output            = output_level,
        TrilinosSolverOptionsNames::_maxIterationNum   = max_iter,
        TrilinosSolverOptionsNames::_stopCriteriaValue = tol,
        TrilinosSolverOptionsNames::_maxRestartIterationNum  = get<int>(options,"tri-max-restart-iteration-num"),
        TrilinosSolverOptionsNames::_preconditioner    = precond_type,
        TrilinosSolverOptionsNames::_solver            = solver_type,
        TrilinosSolverOptionsNames::_useThread         = (get<int>(options,"hts-use-thread") == 1),
        TrilinosSolverOptionsNames::_nbThreads         = get<int>(options,"hts-nb-threads"),
        TrilinosSolverOptionsNames::_iluk              = options_iluk,
        TrilinosSolverOptionsNames::_ilut              = options_ilut,
        TrilinosSolverOptionsNames::_filu              = options_filu,
        TrilinosSolverOptionsNames::_relaxation        = options_relaxation,
        TrilinosSolverOptionsNames::_chebyshev         = options_chebyshev,
        TrilinosSolverOptionsNames::_schwarz           = options_schwarz,
        TrilinosSolverOptionsNames::_muelu             = options_muelu);

    // service
#ifdef KOKKOS_ENABLE_SERIAL
    return  new Alien::TrilinosLinearSolver<BackEnd::tag::tpetraserial>(pm, solver_options);
#else
#ifdef KOKKOS_ENABLE_OPENMP
    return  new Alien::TrilinosLinearSolver<BackEnd::tag::tpetraomp>(pm, solver_options);
#endif
#endif
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

#ifdef KOKKOS_ENABLE_SERIAL
typedef SolverFabric<Alien::BackEnd::tag::tpetraserial> TrilinosSolverFabric ;
#else
#ifdef KOKKOS_ENABLE_OPENMP
typedef SolverFabric<Alien::BackEnd::tag::tpetraomp> TrilinosSolverFabric ;
#endif
#endif
REGISTER_SOLVER_FABRIC(TrilinosSolverFabric);

} // namespace Alien
