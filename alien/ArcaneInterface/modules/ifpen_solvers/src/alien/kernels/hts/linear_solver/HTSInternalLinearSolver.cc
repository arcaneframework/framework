﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>

#include "alien/kernels/hts/HTSPrecomp.h"

#ifdef ALIEN_USE_HARTS
#include "HARTS/HARTS.h"
#endif

#ifdef ALIEN_USE_HTSSOLVER
#include "HARTSSolver/HTS.h"
#include "HARTSSolver/MatrixVector/CSR/CSRProfileImpT.h"
#include "HARTSSolver/MatrixVector/CSR/CSRMatrixImpT.h"
#endif

#include <alien/data/Space.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/kernels/hts/algebra/HTSLinearAlgebra.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/kernels/hts/algebra/HTSLinearAlgebra.h>
#include <alien/kernels/hts/linear_solver/HTSInternalLinearSolver.h>
#include <alien/core/backend/LinearSolverT.h>
#include <alien/core/backend/SolverFabricRegisterer.h>
#include <alien/core/block/ComputeBlockOffsets.h>
#include <ALIEN/axl/HTSSolver_IOptions.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

// Compile HTSLinearSolver.
// template class ALIEN_IFPEN_SOLVERS_EXPORT LinearSolver<BackEnd::tag::htssolver>;

/*---------------------------------------------------------------------------*/
HTSInternalLinearSolver::HTSInternalLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng, IOptionsHTSSolver* options)
: m_parallel_mng(parallel_mng)
, m_options(options)
, m_stater(this)
{
}

void
HTSInternalLinearSolver::init(int argc, char const** argv)
{
#ifdef ALIEN_USE_HTSSOLVER
  m_hts_solver.reset(new HartsSolver::HTSSolver());
#endif
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearSolver::init()
{
  SolverStatSentry<HTSInternalLinearSolver> sentry(m_stater, BaseSolverStater::eInit);
  m_output_level = m_options->output();

#ifdef ALIEN_USE_HTSSOLVER
  m_use_mpi = m_parallel_mng->commSize() > 1;
  m_machine_info.init(m_parallel_mng->commRank() == 0);

  m_hts_solver.reset(new HartsSolver::HTSSolver());
  m_hts_solver->setMachineInfo(&m_machine_info);

  bool use_simd = m_options->useSimd();
  m_runtime_configuration.m_memory_space.m_is_mpi = m_use_mpi;
  m_runtime_configuration.m_execution_space.m_use_simd = use_simd;
  switch (m_options->threadEnvType()) {
  case 0:
    m_runtime_configuration.m_execution_space.m_thread_env_type = HARTS::PTh;
    break;
  case 1:
    m_runtime_configuration.m_execution_space.m_thread_env_type = HARTS::OpenMP;
    break;
  default:
    m_runtime_configuration.m_execution_space.m_thread_env_type = HARTS::PTh;
    break;
  }
#endif
  m_current_ctx_id = m_hts_solver->createNewContext();
  HartsSolver::HTSSolver::ContextType& context =
      m_hts_solver->getContext(m_current_ctx_id);

  typedef HartsSolver::MPIInfo MPIEnvType;
  if (m_use_mpi) {
    auto* mpi_mng =
        dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(m_parallel_mng);
    MPI_Comm comm = *static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());
    MPIEnvType* mpi_env = new HartsSolver::MPIInfo();
    mpi_env->init(comm, false); // external MPI management
    context.set<MPIEnvType>(HartsSolver::HTSSolver::Context::MPIEnv, mpi_env);
  }

  m_hts_solver->setCurrentContext(&context);

  m_hts_solver->setParameter<int>("output", m_output_level);
  m_hts_solver->setParameter<int>("parallel-trace", m_options->parallelTrace());

  if (m_options->normalizeOpt())
    m_hts_solver->setParameter<int>("normalize-opt", 1);
  m_hts_solver->setParameter<int>("max-iteration", m_options->maxIterationNum());
  m_hts_solver->setParameter<double>("tol", m_options->stopCriteriaValue());
  m_hts_solver->setParameter<int>("solver", (int)m_options->solver());
  m_hts_solver->setParameter<int>("precond-type", (int)m_options->preconditioner());
  switch (m_options->preconditioner()) {
  case HTSOptionTypes::Poly:
  case HTSOptionTypes::Chebyshev:
    m_hts_solver->setParameter<double>("poly-factor", m_options->polyFactor());
    m_hts_solver->setParameter<double>(
        "poly-eigenvalue-ratio", m_options->polyEigenvalueRatio());
    m_hts_solver->setParameter<double>(
        "poly-eigenvalue-max", m_options->polyEigenvalueMax());
    m_hts_solver->setParameter<double>(
        "poly-eigenvalue-min", m_options->polyEigenvalueMin());

    m_hts_solver->setParameter<int>("poly-degree", m_options->polyDegree());
    m_hts_solver->setParameter<int>(
        "poly-factor-max-iter", m_options->polyFactorMaxIter());
    break;
  case HTSOptionTypes::ILU0FP:
    m_hts_solver->setParameter<int>("ilufp-factor-niter", m_options->ilufpFactorNiter());
    m_hts_solver->setParameter<int>("ilufp-solver-niter", m_options->ilufpSolverNiter());
    m_hts_solver->setParameter<double>("ilufp-tol", m_options->ilufpTol());
    break;
  case HTSOptionTypes::Cpr:
  case HTSOptionTypes::DDMLPC:
    if (m_options->mlOpt().size() > 0) {
      auto const& opt = m_options->mlOpt()[0];
      m_hts_solver->setParameter<int>("ml-output", opt->output());
      m_hts_solver->setParameter<int>("ml", opt->algo());
      m_hts_solver->setParameter<int>("ml-iter", opt->iter());
      m_hts_solver->setParameter<double>("ml-tol", opt->tol());
      m_hts_solver->setParameter<int>("ml-nev", opt->nev());
      m_hts_solver->setParameter<int>("ml-evtype", opt->evtype());
      m_hts_solver->setParameter<double>("ml-evbound", opt->evbound());
      m_hts_solver->setParameter<double>("ml-evtol", opt->evtol());
      m_hts_solver->setParameter<int>("ml-ev-max-iter", opt->evMaxIter());
      m_hts_solver->setParameter<int>("ml-coarse-op", opt->coarseOp());
      m_hts_solver->setParameter<int>("ml-solver", opt->solver());
      m_hts_solver->setParameter<int>("ml-solver-iter", opt->solverIter());
      m_hts_solver->setParameter<double>("ml-solver-tol", opt->solverTol());
      m_hts_solver->setParameter<int>("ml-solver-nev", opt->solverNev());
      m_hts_solver->setParameter<int>("ml-coarse-solver", opt->coarseSolver());
      m_hts_solver->setParameter<int>("ml-coarse-solver-ntile", opt->coarseSolverNtile());
      m_hts_solver->setParameter<int>("ml-neumann-cor", opt->neumannCor());
      m_hts_solver->setParameter<int>("ilu-level", m_options->iluLevel());
      m_hts_solver->setParameter<double>("ilu-drop-tol", m_options->iluDropTol());
    }
    break;
  case HTSOptionTypes::AMGPC:
  case HTSOptionTypes::CprAMG:
  case HTSOptionTypes::CprDDML:
  default:
    break;
  }
  if (m_options->useThread())
    m_hts_solver->setParameter<int>("use-thread", 1);
  m_hts_solver->setParameter<int>("nb-threads", m_options->nbThreads());
  m_hts_solver->setParameter<int>("nb-part", m_options->nbPart());
  m_hts_solver->setParameter<int>("nb-subpart", m_options->nbSubpart());
  m_hts_solver->setParameter<int>("metis", m_options->metis());
  m_hts_solver->setParameter<int>("smetis", m_options->smetis());
  m_hts_solver->setParameter<int>("sendrecv-opt", m_options->sendrecvOpt());
  m_hts_solver->setParameter<int>("pqueue", m_options->pqueue());
  m_hts_solver->setParameter<int>("affinity-mode", m_options->affinityMode());
}

void
HTSInternalLinearSolver::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  m_parallel_mng = pm;
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearSolver::end()
{
}

#ifdef ALIEN_USE_HTSSOLVER
bool
HTSInternalLinearSolver::solve(
    CSRMatrixType const& A, CSRVectorType const& b, CSRVectorType& x)
{
  using namespace HartsSolver;

  if (m_output_level > 0)
    alien_info([&] { cout() << "HTSLinearSolver::solve"; });

  _startPerfCount();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // GET CURRENT CONTEXT
  //
  HartsSolver::HTSSolver::ContextType& context =
      m_hts_solver->getContext(m_current_ctx_id);
  m_hts_solver->setCurrentContext(&context);

  bool is_parallel = false;
  auto* parallel_mng = A.distribution().parallelMng();
  if (parallel_mng)
    is_parallel = parallel_mng->commSize() > 1;
  typedef HartsSolver::MPIInfo MPIEnvType;
  MPIEnvType* mpi_env = nullptr;
  if (is_parallel) {
    mpi_env = context.get<MPIEnvType>(HTSSolver::Context::MPIEnv);
    if (mpi_env == nullptr) {
      auto* mpi_mng = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(
          m_parallel_mng);
      MPI_Comm comm = *static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());
      MPIEnvType* mpi_env = new HartsSolver::MPIInfo();
      mpi_env->init(comm, false); // external MPI management
      context.set<MPIEnvType>(HTSSolver::Context::MPIEnv, mpi_env);
    }
  }
  ///////////////////////////////////////////////
  //
  // GET CURRENT PROFILE
  //

  typedef HartsSolver::CSRProfile MCProfileType;
  typedef HartsSolver::ProfileView MCProfileViewType;
  typedef MCProfileType::PermutationType MCProfilePermType;

  CSRMatrixType::ProfileType const& matrix_profile = A.internal()->getCSRProfile();
  int nrows = matrix_profile.getNRow();
  int const* kcol = matrix_profile.getRowOffset().unguardedBasePointer();
  int const* cols = matrix_profile.getCols().unguardedBasePointer();

  typedef Graph::MPIPartition MPIPartitionType;
  MPIPartitionType* partition_info =
      context.get<MPIPartitionType>(HTSSolver::Context::MPIPartition);
  if (is_parallel && partition_info == NULL) {
    Graph::MPIPartition* mpi_partition =
        new Graph::MPIPartition(mpi_env->getParallelMng());
    mpi_partition->init(nrows);
    MCProfileViewType profile(kcol, cols, nrows);
    mpi_partition->compute(profile, true);
    partition_info = mpi_partition;
    context.set<MPIPartitionType>(HTSSolver::Context::MPIPartition, partition_info);
  }

  ////////////////////////////////////////////////////
  //
  // SOLVE
  //
  Integer error = 0;
  MCProfileType* profile = context.get<MCProfileType>(HTSSolver::Context::MatrixProfile);
  MCProfilePermType* profile_permutation = NULL;
  if (profile == NULL) {
    if (partition_info) {
      profile_permutation = new MCProfilePermType(*partition_info);
      context.set<MCProfilePermType>(
          HTSSolver::Context::ProfilePermutation, profile_permutation);
    }
    if (profile_permutation) {
      profile = new MCProfileType();
      profile->init(nrows, kcol, cols, *profile_permutation);
    } else
      profile = new MCProfileType(nrows, kcol, cols);
    context.set<MCProfileType>(HTSSolver::Context::MatrixProfile, profile);
  }
  // int block_size =  A.space().structInfo().size() ;
  int block_size = A.block() ? A.block()->size() : 1;
  switch (block_size) {
  case 1: {
    typedef HartsSolver::CSRMatrix<Real, 1> MCMatrixType;
    MCMatrixType matrix(profile);
    matrix.setValues(A.internal()->getDataPtr(), profile_permutation);
    // matrix.updateValues(nrows,kcol,block_size,A.internal()->getDataPtr()) ;
    RunOp op(m_hts_solver.get(), matrix, b.getDataPtr(), x.getDataPtr(), m_hts_status,
        context);
    HARTS::DispatchRun<RunOp>(m_runtime_configuration, op);
  } break;
  case 2: {
    typedef HartsSolver::CSRMatrix<Real, 2> MCMatrixType;
    typedef MCMatrixType::MatrixDataType MatrixDataType;
    typedef MCMatrixType::VectorDataType VectorDataType;
    MCMatrixType matrix(profile);
    matrix.setValues(
        (MatrixDataType const*)(A.internal()->getDataPtr()), profile_permutation);
    // matrix.updateValues(nrows,kcol,block_size,A.internal()->getDataPtr()) ;
    if (is_parallel)
      error = m_hts_solver->solveN<2, true>(matrix,
          reinterpret_cast<VectorDataType const*>(b.getDataPtr()),
          reinterpret_cast<VectorDataType*>(x.getDataPtr()), m_hts_status, context);
    else
      error = m_hts_solver->solveN<2, false>(matrix,
          reinterpret_cast<VectorDataType const*>(b.getDataPtr()),
          reinterpret_cast<VectorDataType*>(x.getDataPtr()), m_hts_status, context);
  } break;
  case 3: {
    typedef HartsSolver::CSRMatrix<Real, 3> MCMatrixType;
    typedef MCMatrixType::MatrixDataType MatrixDataType;
    typedef MCMatrixType::VectorDataType VectorDataType;
    MCMatrixType matrix(profile);
    matrix.setValues(
        (MatrixDataType const*)(A.internal()->getDataPtr()), profile_permutation);
    // matrix.updateValues(nrows,kcol,block_size,A.internal()->getDataPtr()) ;
    if (is_parallel)
      error = m_hts_solver->solveN<3, true>(matrix,
          reinterpret_cast<VectorDataType const*>(b.getDataPtr()),
          reinterpret_cast<VectorDataType*>(x.getDataPtr()), m_hts_status, context);
    else
      error = m_hts_solver->solveN<3, false>(matrix,
          reinterpret_cast<VectorDataType const*>(b.getDataPtr()),
          reinterpret_cast<VectorDataType*>(x.getDataPtr()), m_hts_status, context);
  } break;
  case 4: {
    typedef HartsSolver::CSRMatrix<Real, 4> MCMatrixType;
    typedef MCMatrixType::MatrixDataType MatrixDataType;
    typedef MCMatrixType::VectorDataType VectorDataType;
    MCMatrixType matrix(profile);
    matrix.setValues(
        (MatrixDataType const*)(A.internal()->getDataPtr()), profile_permutation);
    // matrix.updateValues(nrows,kcol,block_size,A.internal()->getDataPtr()) ;
    if (is_parallel)
      error = m_hts_solver->solveN<4, true>(matrix,
          reinterpret_cast<VectorDataType const*>(b.getDataPtr()),
          reinterpret_cast<VectorDataType*>(x.getDataPtr()), m_hts_status, context);
    else
      error = m_hts_solver->solveN<4, false>(matrix,
          reinterpret_cast<VectorDataType const*>(b.getDataPtr()),
          reinterpret_cast<VectorDataType*>(x.getDataPtr()), m_hts_status, context);
  } break;
  default:
    break;
  }

  _endPerfCount();

  ////////////////////////////////////////////////////
  //
  // ANALIZE STATUS
  m_status.residual = m_hts_status.residual;
  m_status.iteration_count = m_hts_status.num_iter;

  if (error == 0) {
    m_status.succeeded = true;
    m_status.error = 0;
    if (m_output_level > 0) {
      alien_info([&] {
        cout() << "Resolution info      :";
        cout() << "Resolution status      : OK";
        cout() << "Residual             : " << m_hts_status.residual;
        cout() << "Number of iterations : " << m_hts_status.num_iter;
      });
    }
    return true;
  } else {
    m_status.succeeded = false;
    m_status.error = m_hts_status.error;
    if (m_output_level > 0) {
      alien_info([&] {
        cout() << "Resolution status      : Error";
        cout() << "Error code             : " << m_hts_status.error;
      });
    }
    return false;
  }
}
#endif

void
HTSInternalLinearSolver::_startPerfCount()
{
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearSolver::_endPerfCount()
{
  m_solve_num++;
  m_total_iter_num += m_status.iteration_count;
}

const Alien::SolverStatus&
HTSInternalLinearSolver::getStatus() const
{
  if (m_output_level > 0) {
    printInfo();
  }
  return m_status;
}

void
HTSInternalLinearSolver::printInfo()
{
  alien_info([&] {
    cout();
    cout() << "|--------------------------------------------|";
    cout() << "| Linear Solver        : HTSSolver           |";
    cout() << "|--------------------------------------------|";
    cout() << "| total solver time    : " << m_total_solve_time;
    cout() << "| total system time    : " << m_total_system_time;
    cout() << "| total num of iter    : " << m_total_iter_num;
    cout() << "| solve num            : " << m_solve_num;
    cout() << "| internal setup time  : " << m_int_total_setup_time;
    cout() << "| internal solve time  : " << m_int_total_solve_time;
    cout() << "| internal finish time : " << m_int_total_finish_time;
    cout() << "|---------------------------------------------|";
    cout();
  });
}

void
HTSInternalLinearSolver::printInfo() const
{
  alien_info([&] {
    cout();
    cout() << "|--------------------------------------------|";
    cout() << "| Linear Solver        : HTSSolver           |";
    cout() << "|--------------------------------------------|";
    cout() << "| total solver time    : " << m_total_solve_time;
    cout() << "| total system time    : " << m_total_system_time;
    cout() << "| total num of iter    : " << m_total_iter_num;
    cout() << "| solve num            : " << m_solve_num;
    cout() << "| internal setup time  : " << m_int_total_setup_time;
    cout() << "| internal solve time  : " << m_int_total_solve_time;
    cout() << "| internal finish time : " << m_int_total_finish_time;
    cout() << "|---------------------------------------------|";
    cout();
  });
}

bool
HTSInternalLinearSolver::solve(IMatrix const& A, IVector const& b, IVector& x)
{
  using namespace Alien;

#ifdef ALIEN_USE_HTSSOLVER
  SolverStatSentry<HTSInternalLinearSolver> sentry(m_stater, BaseSolverStater::ePrepare);
  CSRMatrixType const& matrix = A.impl()->get<BackEnd::tag::simplecsr>();
  CSRVectorType const& rhs = b.impl()->get<BackEnd::tag::simplecsr>();
  CSRVectorType& sol = x.impl()->get<BackEnd::tag::simplecsr>(true);
  sentry.release();

  SolverStatSentry<HTSInternalLinearSolver> sentry2(m_stater, BaseSolverStater::eSolve);
  return solve(matrix, rhs, sol);
#endif
}

std::shared_ptr<ILinearAlgebra>
HTSInternalLinearSolver::algebra() const
{
  return std::shared_ptr<ILinearAlgebra>(new Alien::HTSLinearAlgebra());
}
/*
IInternalLinearSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >*
HTSInternalLinearSolverFactory(IParallelMng* p_mng, IOptionsHTSSolver* options)
{
  return new HTSInternalLinearSolver(p_mng, options);
}*/
ILinearSolver*
HTSInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsHTSSolver* options)
{
  return new HTSInternalLinearSolver(p_mng, options);
}

}

#include <alien/kernels/hts/linear_solver/HTSInternalLinearSolver.h>
#include <alien/kernels/hts/linear_solver/HTSOptionTypes.h>
#include <alien/kernels/hts/linear_solver/arcane/HTSLinearSolver.h>
#include <ALIEN/axl/HTSSolver_axl.h>
#include <ALIEN/axl/HTSSolver_StrongOptions.h>

namespace Alien {

template<>
class SolverFabric<Alien::BackEnd::tag::htssolver>
: public ISolverFabric
{
public :
  BackEndId backend() const {
     return "htssolver" ;
  }

  void
  add_options(CmdLineOptionDescType& cmdline_options) const
  {
    using namespace boost::program_options;
    options_description desc("HTSSolver options");
    desc.add_options()("hts-solver",     value<std::string>()->default_value("bicgs"),"solver algo name : bicgstab ddml")
                      ("hts-precond",    value<std::string>()->default_value("none"),"preconditioner diag none poly chebyshev bssor ilu0 ilu0fp ddml amg cpramg")
                      ("hts-nb-threads", value<int>()->default_value(1), "number of thread for multithreaded solver")
                      ("hts-pqueue",     value<int>()->default_value(0),"Parallel Queue System :\n \t 0->Single\n \t 1->Distributed \n \t 2->Squential")
                      ("hts-thread-env-type",value<int>()->default_value(0),"ThreadEnv System : \n \t 0->Pth\n \t 1->OpenMP \n \t 2->TBB")
                      ("hts-affinity-mode",  value<int>()->default_value(0),"Affinity Mode : \n \t 0->Block \n \t 1->Interleave")
                      ("hts-use-simd",       value<int>()->default_value(0)," enable simd optimization")
                      ("hts-nb-part",        value<int>()->default_value(1),"number of domain partitions")
                      ("hts-nb-subpart",     value<int>()->default_value(0),"number of subdomain partitions")
                      ("hts-metis",          value<int>()->default_value(1),"to use metis partitioner on each MPI domains")
                      ("hts-smetis",         value<int>()->default_value(1),"to use metis partitioner on each MPI domains")
                      ("hts-poly-factor",    value<double>()->default_value(0.), "polynomial factor")
                      ("hts-poly-eigenvalue-ratio",value<double>()->default_value(30.),"polynomial eigen ratio")
                      ("hts-poly-eigenvalue-max",  value<double>()->default_value(0.),"polynomial eigenvalue max factor")
                      ("hts-poly-eigenvalue-min",  value<double>()->default_value(0.),"polynomial eigenvalue min factor")
                      ("hts-poly-factor-max-iter", value<int>()->default_value(3),"polynomial max iter factor")
                      ("hts-poly-degree",          value<int>()->default_value(3),"polynomial degree")
                      ("hts-ilufp-factor-niter",   value<int>()->default_value(0),"fixed point ilu number of factorization iterations")
                      ("hts-ilufp-solver-niter",   value<int>()->default_value(1),"fixed point ilu number of solver iterations")
                      ("hts-ilufp-tol",            value<double>()->default_value(0.),"fixed point ilu tolerance")
                      ("hts-ilu-level",            value<int>()->default_value(0),"iluk level")
                      ("hts-ilu-drop-tol",         value<double>()->default_value(0.),"iluk drop tolerance")
                      ("hts-ml-algo",              value<int>()->default_value(0),"0->AS, 1->ML")
                      ("hts-ml-iter",              value<int>()->default_value(3),"ML iter")
                      ("hts-ml-tol",               value<double>()->default_value(0.5),"ML tolerance")
                      ("hts-ml-nev",               value<int>()->default_value(1),"ML nb max of eigen values")
                      ("hts-ml-evtype",            value<int>()->default_value(1),"ML Eigen Solver type : \n 0->SLEPC \n \t 1->ARPACK \n \t 2->Spectra")
                      ("hts-ml-evbound",           value<double>()->default_value(0.), "nb max of eigen values")
                      ("hts-ml-evtol",             value<double>()->default_value(1.e-6),"ML ev algo tolerance")
                      ("hts-ml-ev-max-iter",       value<int>()->default_value(1000),"ML ev algo max iter")
                      ("hts-ml-coarse-op",         value<int>()->default_value(1),"ML option -- Coarse operator choice -- \n \t 1) Nicolaides \n \t  2) GenEO")
                      ("hts-ml-solver",            value<int>()->default_value(0),"DDML option -- local solver choice -- \n \t 0->LU, \n \t 1->LUS, \n \t 2->BCGS, \n \t 3->ILUBCGS")
                      ("hts-ml-solver-iter",       value<int>()->default_value(100),"ML local solver iter")
                      ("hts-ml-solver-tol",        value<double>()->default_value(1.e-6),"ML local solver tolerance")
                      ("hts-ml-solver-nev",        value<int>()->default_value(1),"ML local solver nb max of eigen values")
                      ("hts-ml-coarse-solver",     value<int>()->default_value(0),"DDML option -- coarse solver choice -- \n \t 0->LU, \n \t 1->LUS, \n \t 2->LUMT, \n \t 3->LUMTS \n \t 4->DistLU")
                      ("hts-ml-coarse-solver-ntile",  value<int>()->default_value(1), "nb domain per tile")
                      ("hts-ml-neumann-cor",          value<int>()->default_value(-1),"ML Neumann cor")
                      ("hts-relax-solver",         value<int>()->default_value(0),"relax solver option")
                      ("hts-cpr-solver",           value<int>()->default_value(0),"cpr solver option")
                      ("hts-amg-algo",             value<std::string>()->default_value("PMIS"),"AMG algorithm option, AGGREGATION or PMIS");
    cmdline_options.add(desc) ;
  }


  template<typename OptionT>
  Alien::ILinearSolver* _create(OptionT const& options,Alien::IMessagePassingMng* pm) const
  {
    int output_level = get<int>(options,   "output-level") ;
    double tol       = get<double>(options,"tol");
    int max_iter     = get<int>(options,   "max-iter");

    std::string precond_type_s = get<std::string>(options,"hts-precond");
    HTSOptionTypes::ePreconditioner precond_type =
        OptionsHTSSolverUtils::stringToPreconditionerEnum(precond_type_s);

    std::string solver_type_s = get<std::string>(options,"hts-solver");
    HTSOptionTypes::eSolver solver_type =
        OptionsHTSSolverUtils::stringToSolverEnum(solver_type_s);

    auto options_ml = std::make_shared<HTSSolverOptionsNames::StrongOptionsMLOptType>(
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_output            = get<int>(options,"hts-ml-output"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_algo              = get<int>(options,"hts-ml-algo"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_iter              = get<int>(options,"hts-ml-iter"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_tol               = get<double>(options,"hts-ml-tol"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_nev               = get<int>(options,"hts-ml-nev"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_evtype            = get<int>(options,"hts-ml-evtype"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_evbound           = get<int>(options,"hts-ml-evbound"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_evtol             = get<double>(options,"hts-ml-evtol"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_evMaxIter         = get<int>(options,"hts-ml-ev-max-iter"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_coarseOp          = get<int>(options,"hts-ml-coarse-op"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_solver            = get<int>(options,"hts-ml-solver"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_solverIter        = get<int>(options,"hts-ml-solver-iter"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_solverTol         = get<double>(options,"hts-ml-solver-tol"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_solverNev         = get<int>(options,"hts-ml-solver-nev"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_coarseSolver      = get<int>(options,"hts-ml-coarse-solver"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_coarseSolverNtile = get<int>(options,"hts-ml-coarse-solver-ntile"),
        HTSSolverOptionsNames::MLOptTypeOptionsNames::_neumannCor        = get<int>(options,"hts-ml-neumann-cor"));

    // options
    auto solver_options = std::make_shared<StrongOptionsHTSSolver>(
        HTSSolverOptionsNames::_output            = output_level,
        HTSSolverOptionsNames::_maxIterationNum   = max_iter,
        HTSSolverOptionsNames::_stopCriteriaValue = tol,
        HTSSolverOptionsNames::_preconditioner    = precond_type,
        HTSSolverOptionsNames::_solver            = solver_type,
        HTSSolverOptionsNames::_polyFactor          = get<double>(options,"hts-poly-factor"),
        HTSSolverOptionsNames::_polyEigenvalueRatio = get<double>(options,"hts-poly-eigenvalue-ratio"),
        HTSSolverOptionsNames::_polyEigenvalueMax   = get<double>(options,"hts-poly-eigenvalue-max"),
        HTSSolverOptionsNames::_polyEigenvalueMin   = get<double>(options,"hts-poly-eigenvalue-min"),
        HTSSolverOptionsNames::_polyFactorMaxIter   = get<int>(options,"hts-poly-factor-max-iter"),
        HTSSolverOptionsNames::_polyDegree         = get<int>(options,"hts-poly-degree"),
        HTSSolverOptionsNames::_ilufpFactorNiter  = get<int>(options,"hts-ilufp-factor-niter"),
        HTSSolverOptionsNames::_ilufpSolverNiter  = get<int>(options,"hts-ilufp-solver-niter"),
        HTSSolverOptionsNames::_ilufpTol          = get<double>(options,"hts-ilufp-tol"),
        HTSSolverOptionsNames::_iluLevel          = get<int>(options,"hts-ilu-level"),
        HTSSolverOptionsNames::_iluDropTol        = get<double>(options,"hts-ilu-drop-tol"),
        HTSSolverOptionsNames::_useUnitDiag       = (get<int>(options,"hts-use-unit-diag") == 1),
        HTSSolverOptionsNames::_keepDiagOpt       = (get<int>(options,"hts-keep-diag-opt")== 1),
        HTSSolverOptionsNames::_reorderOpt        = (get<int>(options,"hts-reorder-opt") == 1),
        HTSSolverOptionsNames::_interfaceOpt      = (get<int>(options,"hts-interface-opt") == 1),
        HTSSolverOptionsNames::_relaxSolver       = get<int>(options,"hts-relax-solver"),
        HTSSolverOptionsNames::_cprSolver         = get<int>(options,"hts-cpr-solver"),
        HTSSolverOptionsNames::_amgAlgo           = get<std::string>(options,"hts-amg-algo"),
        HTSSolverOptionsNames::_normalizeOpt      = (get<int>(options,"hts-normalize-opt") == 1),
        HTSSolverOptionsNames::_useThread         = (get<int>(options,"hts-use-thread") == 1),
        HTSSolverOptionsNames::_nbThreads         = get<int>(options,"hts-nb-threads"),
        HTSSolverOptionsNames::_pqueue            = get<int>(options,"hts-pqueue"),
        HTSSolverOptionsNames::_threadEnvType     = get<int>(options,"hts-thread-env-type"),
        HTSSolverOptionsNames::_affinityMode      = get<int>(options,"hts-affinity-mode"),
        HTSSolverOptionsNames::_useSimd           = (get<int>(options,"hts-use-simd")==1),
        HTSSolverOptionsNames::_nbPart            = get<int>(options,"hts-nb-part"),
        HTSSolverOptionsNames::_nbSubpart         = get<int>(options,"hts-nb-subpart"),
        HTSSolverOptionsNames::_metis             = get<int>(options,"hts-metis"),
        HTSSolverOptionsNames::_smetis            = get<int>(options,"hts-smetis"),
        HTSSolverOptionsNames::_sendrecvOpt       = get<int>(options,"hts-sendrecv-opt"),
        HTSSolverOptionsNames::_dumpMatFileName   = get<std::string>(options,"hts-dump-mat-file-name"),
        HTSSolverOptionsNames::_mlOpt             = options_ml);
    // service
    return  new Alien::HTSLinearSolver(pm, solver_options);
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

typedef SolverFabric<Alien::BackEnd::tag::htssolver> HTSSOLVERSolverFabric ;
REGISTER_SOLVER_FABRIC(HTSSOLVERSolverFabric);

} // namespace Alien
