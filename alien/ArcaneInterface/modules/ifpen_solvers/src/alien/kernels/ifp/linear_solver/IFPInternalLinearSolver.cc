// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/kernels/ifp/linear_solver/IFPInternalLinearSolver.h>

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/core/backend/LinearSolverT.h>
#include <alien/kernels/ifp/linear_solver/IFPSolverProperty.h>

#include <alien/core/backend/SolverFabricRegisterer.h>
/*---------------------------------------------------------------------------*/

#include "IFPSolverProperty.h"
#include "IFPSolver.h"

#include <alien/utils/Trace.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/ifp/data_structure/IFPVector.h>
#include <alien/kernels/ifp/data_structure/IFPMatrix.h>
#include <alien/kernels/ifp/IFPSolverBackEnd.h>
#include <alien/kernels/ifp/data_structure/IFPSolverInternal.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>
#include <ALIEN/axl/IFPLinearSolver_IOptions.h>
#include <alien/data/Space.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include "mpi.h"

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
IFPInternalLinearSolver::IFPInternalLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    IOptionsIFPLinearSolver* options)
: m_parallel_mng(parallel_mng)
, m_print_info(0)
, m_options(options)
, m_stater(this)
{
}

/*---------------------------------------------------------------------------*/

IFPInternalLinearSolver::~IFPInternalLinearSolver()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
IFPInternalLinearSolver::init()
{
  SolverStatSentry<IFPInternalLinearSolver> sentry(m_stater, BaseSolverStater::eInit);
  if (m_parallel_mng == nullptr)
    return;

  alien_info(m_print_info, [&] { cout() << "IFPInternalLinearSolver::init"; });

  if (m_options->output() > 0)
    m_print_info = m_options->output();
  else if (m_options->verbose())
    m_print_info = 1;

  if (m_parallel_mng->commRank() == 0) {
    F2C(ifpsolversetoutputlevel)(&m_print_info);
  } else {
    Integer print_info = 0;
    F2C(ifpsolversetoutputlevel)
    (&print_info); // IFPSolve should not keep a reference to print_info
  }

  m_max_iteration = m_options->numIterationsMax();
  m_stop_criteria_value = m_options->stopCriteriaValue();
  switch (m_options->precondOption()) {
  case IFPSolverProperty::Diag:
    m_precond_option = 0;
    break;
  case IFPSolverProperty::ILU0:
    m_precond_option = 1;
    break;
  case IFPSolverProperty::AMG:
    m_precond_option = 2;
    break;
  case IFPSolverProperty::CprAmg:
    m_precond_option = 3;
    break;
  default:
    m_precond_option = 0;
  }

  switch (m_options->ilu0Algo()) {
  case IFPSolverProperty::Normal:
    m_ilu0_algo = 0;
    break;
  case IFPSolverProperty::BlockJacobi:
    m_ilu0_algo = 1;
    break;
  case IFPSolverProperty::Optimized:
    m_ilu0_algo = 2;
    break;
  default:
    m_ilu0_algo = 0;
  }

  m_precond_pressure = (m_options->precondEquation() == IFPSolverProperty::Pressure);
  m_normalisation_pivot = m_options->normalisationPivot();
  m_normalize_opt = m_options->needNormalisation();
  m_keep_rhs = m_options->keepRhs();

  Integer needMpiInit = 0;
  Integer fcomm = 0;

  {
    auto mpi_mng =
        dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(m_parallel_mng);
    const void* ptr = nullptr ;
    if(mpi_mng)
       ptr = mpi_mng->getMPIComm();
    if (ptr) {
      auto* comm = static_cast<const MPI_Comm*>(ptr);
      fcomm = MPI_Comm_c2f(*comm);
      needMpiInit = 1;
    } else {
      fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);
      needMpiInit = 0;
    }
  }

  F2C(ifpsolverinit)(&fcomm, &needMpiInit, &m_normalize_opt);
}

/*---------------------------------------------------------------------------*/

void
IFPInternalLinearSolver::setNullSpaceConstantOption(bool flag)
{
  alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
}

/*---------------------------------------------------------------------------*/

void
IFPInternalLinearSolver::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  m_parallel_mng = pm;
}

/*---------------------------------------------------------------------------*/

void
IFPInternalLinearSolver::end()
{
  {
    F2C(ifpsolverfreedata)();
  }
  internalPrintInfo();
}

/*---------------------------------------------------------------------------*/

const Alien::SolverStatus&
IFPInternalLinearSolver::getStatus() const
{
  return m_status;
}

/*---------------------------------------------------------------------------*/

bool
IFPInternalLinearSolver::solve(const MatrixType& A, const VectorType& b, VectorSolType& x)
{
  alien_info(m_print_info, [&] {
    cout() << "|--------------------------------------------------------|";
    cout() << "| Start Linear Solver #" << m_stat.solveCount();
  });

  if (m_parallel_mng == nullptr)
    return true;

  bool isSolverOk = false;

  SolverStatSentry<IFPInternalLinearSolver> sentry(m_stater, BaseSolverStater::ePrepare);

  using namespace Alien;
  // C'est a ce moment la, que la IFP matrice est construite.
  // Avant d'arriver la, la matrice est stockee dans la SimpleCSRMatrix
  // la conversion de representation se fait a ce niveau là
  const IFPMatrix& matrix = A.impl()->get<BackEnd::tag::ifpsolver>();
  if (m_keep_rhs)
    Alien::IFPSolverInternal::VectorInternal::initRHS(true);
  const IFPVector& rhs = b.impl()->get<BackEnd::tag::ifpsolver>();
  if (m_keep_rhs) {
    Alien::IFPSolverInternal::VectorInternal::initRHS(false);
    x.impl()->get<BackEnd::tag::ifpsolver>();
  } else
    b.impl()->get<BackEnd::tag::ifpsolver>();

  alien_fatal((not(matrix.internal()->m_filled and rhs.internal()->m_filled)), [&] {
    cout() << "IFPSolver initialization error "
           << " A: " << matrix.internal()->m_filled
           << " rhs: " << rhs.internal()->m_filled;
  });

  alien_info((m_print_info and matrix.internal()->m_elliptic_split_tag),
      [&] { cout() << "Elliptic split tag enabled"; });

  if (!matrix.getSymmetricProfile() && m_options->ilu0Algo() == IFPSolverProperty::Normal)
    m_ilu0_algo = 3;
  if (!matrix.getSymmetricProfile()
      && m_options->ilu0Algo() == IFPSolverProperty::Optimized)
    m_ilu0_algo = 4;

  sentry.release();


  SolverStatSentry<IFPInternalLinearSolver> sentry2(m_stater, BaseSolverStater::eSolve);

  if (matrix.internal()->m_system_is_resizeable == true)
    isSolverOk = _solveRs(matrix.internal()->m_system_is_resizeable);
  else
    isSolverOk = _solve();

  if (isSolverOk) {
    Alien::IFPSolverInternal::VectorInternal::setRepresentationSwitch(true);
    IFPVector& sol = x.impl()->get<BackEnd::tag::ifpsolver>(true);
    sol.setResizable(rhs.isResizable());
    Alien::IFPSolverInternal::VectorInternal::setRepresentationSwitch(false);
    sol.internal()->m_filled = true;
  } else {
    // La valeur de b en cas d'échec n'est actuellement pas bien définie : à tester.
  }

  sentry2.release();

  if(m_print_info)
    internalPrintInfo();

  alien_info(m_print_info, [&] {
    cout() << "| End Linear Solver";
    cout() << "|--------------------------------------------------------|";
  });

  return isSolverOk;
}

/*---------------------------------------------------------------------------*/

bool
IFPInternalLinearSolver::_solve()
{
  bool resizeable = false;

  F2C(ifpsolversolve)
  (&m_max_iteration, &m_stop_criteria_value, &m_precond_option, &m_precond_pressure,
      &m_normalisation_pivot, &m_ilu0_algo, &resizeable);
  F2C(ifpsolvergetsolverstatus)
  (&m_status.error, &m_status.iteration_count, &m_status.residual);
  m_status.succeeded = (m_status.error == 0);

  return m_status.succeeded;
}

// A. Anciaux
bool
IFPInternalLinearSolver::_solveRs(bool resizeable)
{
  F2C(ifpsolversolve)
  (&m_max_iteration, &m_stop_criteria_value, &m_precond_option, &m_precond_pressure,
      &m_normalisation_pivot, &m_ilu0_algo, &resizeable);
  F2C(ifpsolvergetsolverstatus)
  (&m_status.error, &m_status.iteration_count, &m_status.residual);
  m_status.succeeded = (m_status.error == 0);

  return m_status.succeeded;
}

/*---------------------------------------------------------------------------*/

void
IFPInternalLinearSolver::internalPrintInfo() const
{
  m_stat.print(
      Universe().traceMng(), m_status, String("Linear Solver : IFPLinearSolver"));
  Real init_solver_count = 0;
  Real init_precond_count = 0;
  Real normalize_count = 0;
  Real loop_solver_count = 0;
  F2C(ifpsolvergetperfcount)
  (&init_solver_count, &init_precond_count, &normalize_count, &loop_solver_count);

  // TODO: find a way to automatically set spaces
  alien_info([&]{
    cout() <<     "|--------------------------------------------------------|\n"
    "              | IFPSolver             :                                |\n"
    "              |--------------------------------------------------------|\n"
    "              | init solver time      : " << Arccore::Trace::Precision(4,init_solver_count,true) << "\n"
    "              | init precond time     : " << Arccore::Trace::Precision(4,init_precond_count,true) << "\n"
    "              | normalisation time    : " << Arccore::Trace::Precision(4, normalize_count,true) << "\n"
    "              | loop solver time      : " << Arccore::Trace::Precision(4,loop_solver_count,true) << "\n"
    "              |--------------------------------------------------------|";
  });
}

/*---------------------------------------------------------------------------*/

ILinearSolver*
IFPInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsIFPLinearSolver* options)
{
  return new IFPInternalLinearSolver(p_mng, options);
}

}

#include <alien/kernels/ifp/linear_solver/arcane/IFPLinearSolverService.h>
#include <alien/kernels/ifp/linear_solver/IFPSolverProperty.h>
#include <ALIEN/axl/IFPLinearSolver_IOptions.h>
#include <ALIEN/axl/IFPLinearSolver_StrongOptions.h>

namespace Alien {

template<>
class SolverFabric<Alien::BackEnd::tag::ifpsolver>
: public ISolverFabric
{
public :
  BackEndId backend() const {
     return "ifpsolver" ;
  }

  void
  add_options(CmdLineOptionDescType& cmdline_options) const
  {
    using namespace boost::program_options;
    options_description desc("IFPSolver options");
    desc.add_options()("ifps-solver", value<std::string>()->default_value("bicgs"),"solver algo name : bicgs lu")
        ("ifps-precond", value<std::string>()->default_value("none"),"preconditioner diag none ilu0 amg cpramg");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  Alien::ILinearSolver* _create(OptionT const& options,Alien::IMessagePassingMng* pm) const
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

  Alien::ILinearSolver* create(CmdLineOptionType const& options,Alien::IMessagePassingMng* pm) const
  {
    return _create(options,pm) ;
  }

  Alien::ILinearSolver* create(JsonOptionType const& options,Alien::IMessagePassingMng* pm) const
  {
    return _create(options,pm) ;
  }
};

typedef SolverFabric<Alien::BackEnd::tag::ifpsolver> IFPSOLVERSolverFabric ;
REGISTER_SOLVER_FABRIC(IFPSOLVERSolverFabric);
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
