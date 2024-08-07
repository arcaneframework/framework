﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"
#include <tuple>
#include <iomanip>
#include <regex>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include <alien/data/Space.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/core/backend/LinearSolverT.h>
#include <alien/core/block/ComputeBlockOffsets.h>
#include <alien/kernels/mcg/linear_solver/MCGOptionTypes.h>
#include <alien/kernels/mcg/linear_solver/MCGInternalLinearSolver.h>
#include <alien/kernels/mcg/algebra/MCGInternalLinearAlgebra.h>

#include <Precond/PrecondOptionsEnum.h>
#include <Solvers/AMG/AMGProperty.h>
#include <Solvers/SolverProperty.h>
#include <Solvers/Krylov/BiCGStabDef.h>
#include <MCGSolver/Status.h>
#include <MCGSolver/SolverOptionsEnum.h>
#include <MCGSolver/ILinearSolver.h>
#include <Common/Utils/Machine/MachineInfo.h>
#include <Common/Utils/ParallelEnv.h>

#include "ALIEN/axl/MCGSolver_IOptions.h"

#include <Common/Utils/Machine/MachineInfo.h>

namespace Alien {

std::unique_ptr<MCGInternalLinearSolver::AlienKOpt2MCGKOpt>
    MCGInternalLinearSolver::AlienKOpt2MCGKOpt::m_instance;

MCGInternalLinearSolver::AlienKOpt2MCGKOpt::AlienKOpt2MCGKOpt()
{
  m_option_translate[{ MCGOptionTypes::CPU_CBLAS_BCSR, false, false }] =
      MCGSolver::CPUCBLAS;
  m_option_translate[{ MCGOptionTypes::CPU_CBLAS_BCSR, true, false }] =
      MCGSolver::MPI_CPUCBLAS;
  m_option_translate[{ MCGOptionTypes::CPU_CBLAS_BCSR, false, true }] =
      MCGSolver::OMP_CPUCBLAS;
  m_option_translate[{ MCGOptionTypes::CPU_CBLAS_BCSR, true, true }] =
      MCGSolver::MPI_OMP_CPUCBLAS;

  m_option_translate[{ MCGOptionTypes::CPU_AVX_BCSR, false, false }] =
      MCGSolver::CPUAVX;
  m_option_translate[{ MCGOptionTypes::CPU_AVX_BCSR, true, false }] =
      MCGSolver::MPI_CPUAVX;
  m_option_translate[{ MCGOptionTypes::CPU_AVX_BCSR, false, true }] =
      MCGSolver::OMP_CPUAVX;
  m_option_translate[{ MCGOptionTypes::CPU_AVX_BCSR, true, true }] =
      MCGSolver::MPI_OMP_CPUAVX;

  m_option_translate[{ MCGOptionTypes::CPU_AVX2_BCSP, false, false }] =
      MCGSolver::CPUAVX2;
  m_option_translate[{ MCGOptionTypes::CPU_AVX2_BCSP, true, false }] =
      MCGSolver::MPI_CPUAVX2;
  m_option_translate[{ MCGOptionTypes::CPU_AVX2_BCSP, false, true }] =
      MCGSolver::OMP_CPUAVX2;
  m_option_translate[{ MCGOptionTypes::CPU_AVX2_BCSP, true, true }] =
      MCGSolver::MPI_OMP_CPUAVX2;

  m_option_translate[{ MCGOptionTypes::CPU_AVX512_BCSP, false, false }] =
      MCGSolver::CPUAVX512;
  m_option_translate[{ MCGOptionTypes::CPU_AVX512_BCSP, true, false }] =
      MCGSolver::MPI_CPUAVX512;
  m_option_translate[{ MCGOptionTypes::CPU_AVX512_BCSP, false, true }] =
      MCGSolver::OMP_CPUAVX512;
  m_option_translate[{ MCGOptionTypes::CPU_AVX512_BCSP, true, true }] =
      MCGSolver::MPI_OMP_CPUAVX512;

  m_option_translate[{ MCGOptionTypes::GPU_CUBLAS_BELL, false, false }] =
      MCGSolver::GPUCUBLASBELLSpmv;
  m_option_translate[{ MCGOptionTypes::GPU_CUBLAS_BELL, true, false }] =
      MCGSolver::MPI_GPUCUBLASBELLSpmv;

  m_option_translate[{ MCGOptionTypes::GPU_CUBLAS_BCSP, false, false }] =
      MCGSolver::GPUCUBLASBCSPSpmv;
  m_option_translate[{ MCGOptionTypes::GPU_CUBLAS_BCSP, true, false }] =
      MCGSolver::MPI_GPUCUBLASBCSPSpmv;
}

MCGSolver::eKernelType
MCGInternalLinearSolver::AlienKOpt2MCGKOpt::getKernelOption(
    const KConfigType& kernel_config)
{
  if (!m_instance) {
    m_instance.reset(new AlienKOpt2MCGKOpt);
  }

  const auto& p = m_instance->m_option_translate.find(kernel_config);

  if (p != m_instance->m_option_translate.cend()) {
    return p->second;
  } else {
    m_instance->alien_fatal([&] { m_instance->cout() << "Unknow kernel configuration"; });
    throw FatalErrorException(__PRETTY_FUNCTION__); // just build for warning
  }
}

/*---------------------------------------------------------------------------*/
MCGInternalLinearSolver::MCGInternalLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng, IOptionsMCGSolver* options)
: m_parallel_mng(parallel_mng)
, m_options(options)
{
  m_dir_enum[std::string("+Z")] = *((int*)"+Z-Z");
  m_dir_enum[std::string("-Z")] = *((int*)"-Z+Z");
  m_dir_enum[std::string("+X")] = *((int*)"+X-X");
  m_dir_enum[std::string("-X")] = *((int*)"-X+X");
  m_dir_enum[std::string("+Y")] = *((int*)"+Y-Y");
  m_dir_enum[std::string("-Y")] = *((int*)"-Y+Y");
  m_dir_enum[std::string("+D")] = *((int*)"+D-D");
  m_dir_enum[std::string("-D")] = *((int*)"-D+D");

  // check version
  const std::string expected_version("v2.5");
  const std::regex expected_revision_regex("^" + expected_version + ".*");
  m_version = MCGSolver::ILinearSolver<MCGSolver::LinearSolver>::getRevision();

  if(!std::regex_match(m_version,expected_revision_regex))
  {
    alien_info([&]
        { cout()<<"MCGSolver version mismatch: expect " << expected_version << " get " << m_version ; });
  }

  const std::regex end_regex("-[[:alnum:]]+$");
  m_version = std::regex_replace(m_version,end_regex,"");
}

/*---------------------------------------------------------------------------*/

MCGInternalLinearSolver::~MCGInternalLinearSolver()
{
  delete m_system;
  delete m_solver;
  delete m_machine_info;
  delete m_mpi_info;
  delete m_part_info;
#if 0
  if(m_logger)
    m_logger->report();
#endif
}
/*---------------------------------------------------------------------------*/

std::shared_ptr<Alien::ILinearAlgebra>
MCGInternalLinearSolver::algebra() const
{
  return std::shared_ptr<Alien::ILinearAlgebra>(new Alien::MCGInternalLinearAlgebra());
}
/*---------------------------------------------------------------------------*/

void
MCGInternalLinearSolver::init()
{
#if 0
  if(m_options->logger().size() && m_logger==nullptr)
    m_logger.reset(m_options->logger()[0]);
  if(m_logger)
  {
    m_logger->log("package",this->getBackEndName());
    m_logger->start(eStep::init);
    std::ostringstream oss;
    oss << m_options->maxIterationNum();
    m_logger->log("max-it",oss.str());
    oss.str(std::string());
    oss << m_options->stopCriteriaValue();
    m_logger->log("tol",oss.str());
  }
#endif
  m_init_timer.start();

  if (m_parallel_mng == nullptr)
    return;

  m_output_level = m_options->output();
  alien_info([&]{
    cout() << "MCGSolver version " << MCGSolver::ILinearSolver<MCGSolver::LinearSolver>::getRevision();
  });

  m_use_mpi = m_parallel_mng->commSize() > 1;

  auto mpi_mng =
      dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(m_parallel_mng);

  m_machine_info = new MCGSolver::MachineInfo;
  m_machine_info->init(m_parallel_mng->commRank() == 0);

  if (m_use_mpi) {
    MPI_Comm comm = *static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());
    m_mpi_info = new mpi::MPIInfo;
    m_mpi_info->init(comm, false);
  }
  m_use_thread = m_options->useThread();

  m_max_iteration = m_options->maxIterationNum();
  m_precision = m_options->stopCriteriaValue();
  m_precond_opt = m_options->preconditioner();

  m_solver_opt = m_options->solver();

  m_solver = MCGSolver::ILinearSolver<MCGSolver::LinearSolver>::create();

  m_solver->setMachineInfo(m_machine_info);
  if (m_use_mpi)
    m_solver->initMPIInfo(m_mpi_info);

  m_solver->setOpt(MCGSolver::Normalize, m_options->normalize());
  m_solver->setOpt(MCGSolver::OutputLevel, m_output_level - 1);
  m_solver->setOpt(MCGSolver::SolverMaxIter, m_max_iteration);
  m_solver->setOpt(MCGSolver::SolverEps, m_precision);
  if (m_use_thread) {
    const char* env_num_thread = getenv("OMP_NUM_THREADS");
    if (env_num_thread != nullptr) {
      m_num_thread = std::atoi(env_num_thread);
      alien_debug([&]{
        cout() << "Alien MCGSolver: set num_thread to " << m_num_thread << " from env";
      });
    }
    m_solver->setOpt(MCGSolver::UseOmpThread, m_use_thread);
    m_solver->setOpt(MCGSolver::NumThread, m_num_thread);
    m_solver->setOpt(MCGSolver::SharedMemPart, MCGSolver::Graph::Partitioner::METIS_KWAY);
  }

  if (m_options->exportSystem())
    m_solver->setOpt(MCGSolver::ExportSystemFileName,
        std::string(localstr(m_options->exportSystemFileName())));

  m_solver->setOpt(MCGSolver::SolverType, m_solver_opt);

  auto bj_local_solver = m_options->bjLocalPrecond();
  if (m_use_mpi) {
    switch (m_precond_opt) {
    case MCGSolver::PrecColorBlockILU0:
    case MCGSolver::PrecBlockILU0:
    case MCGSolver::PrecParILU0:
    case MCGSolver::PrecILUk:
    case MCGSolver::PrecFixPointILU0:
      bj_local_solver = m_precond_opt;
      m_precond_opt = MCGSolver::PrecBlockJacobi;
    }
  } else {
    if (m_precond_opt == MCGSolver::PrecBlockJacobi) {
      m_precond_opt = bj_local_solver;
    }
  }

  m_solver->setOpt(MCGSolver::PrecondOpt, m_precond_opt);

  m_solver->setOpt(MCGSolver::BlockJacobiNumOfIter, m_options->bjNumIter());
  m_solver->setOpt(MCGSolver::BlockJacobiLocalSolver, bj_local_solver);

  m_solver->setOpt(MCGSolver::FPILUSolverNumIter, m_options->fpilu0SolveNumIter());
  m_solver->setOpt(MCGSolver::FPILUFactorNumIter, m_options->fpilu0FactoNumIter());

  m_solver->setOpt(MCGSolver::SpPrec, m_options->spPrec());

  if (!m_options->Poly().empty()) {
    m_solver->setOpt(MCGSolver::PolyOrder, m_options->Poly()[0]->order());
    m_solver->setOpt(MCGSolver::PolyFactor, m_options->Poly()[0]->factor());
    m_solver->setOpt(MCGSolver::PolyFactorMaxIter, m_options->Poly()[0]->factorNumIter());
  }

  if (!m_options->ILUk().empty()) {
    m_solver->setOpt(MCGSolver::ILUkLevel, m_options->ILUk()[0]->levelOfFill());
    // override spPrec
    m_solver->setOpt(MCGSolver::SpPrec, m_options->ILUk()[0]->sp());
  }

  if (!m_options->CprAmg().empty()) {
    m_solver->setOpt(MCGSolver::CxrSolver, m_options->CprAmg()[0]->cxrSolver());
    m_solver->setOpt(MCGSolver::RelaxSolver, m_options->CprAmg()[0]->relaxSolver());
  }

  if (!m_options->amgx().empty()) {
    m_solver->setOpt(MCGSolver::AmgXConfigFile,
        std::string(localstr(m_options->amgx()[0]->parameterFile())));
    if (m_options->amgx()[0]->parameterFile().empty())
      m_solver->setOpt(MCGSolver::AmgAlgo, m_options->amgx()[0]->amgAlgo());
    else
      alien_info([&]{
        cout() << "Only parameter-file option is considered";
      });
  }

  m_solver->setOpt(MCGSolver::BiCGStabRhoInit, MCGSolver::RhoInit::RhsSquareNorm);
  m_solver->init(AlienKOpt2MCGKOpt::getKernelOption(
      { m_options->kernel(), m_use_mpi, m_use_thread }));

  m_init_timer.stop();

#if 0
  if(m_logger)
  {
    m_logger->log("precond",OptionsMCGSolverUtils::preconditionerEnumToString(m_precond_opt));
    m_logger->log("solver",OptionsMCGSolverUtils::solverEnumToString(m_solver_opt));
    m_logger->stop(eStep::init);
  }
#endif
}

void
MCGInternalLinearSolver::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  // TODO: do we really want to do that ?
  m_parallel_mng = pm;
}

/*---------------------------------------------------------------------------*/

void
MCGInternalLinearSolver::end()
{}

Integer
MCGInternalLinearSolver::_solve(const MCGMatrixType& A, const MCGVectorType& b,
    MCGVectorType& x, MCGSolver::PartitionInfo* part_info)
{
  alien_debug([&]{
    cout() << "MCGInternalLinearSolver::_solve A:" << A.m_matrix[0][0].get()
           << " b:" << &b << " x:" << &x;
  });

  Integer error = -1;

  m_system_timer.start();
  if (_matrixChanged(A)) {
    delete m_system;
    _registerKey(A, b, x);
    m_system = _createSystem(A, b, x, part_info);

    if (A.m_elliptic_split_tag) {
      m_system->setEquationType(A.m_equation_type);
    }
  } else {
    if (_rhsChanged(b)) {
      m_b_key = b.m_key;
      m_system->setRhs(&b.m_bvector);
    }
  }

  m_solver->setMatrixUpdate(m_A_update);
  m_solver->setRhsUpdate(m_b_update);
  m_system_timer.stop();

  m_solve_timer.start();
  error = m_solver->solve(m_system->getImpl(), &m_mcg_status);
  m_solve_timer.stop();

  return error;
}

Integer
MCGInternalLinearSolver::_solve(const MCGMatrixType& A, const MCGVectorType& b,
    const MCGVectorType& x0, MCGVectorType& x, MCGSolver::PartitionInfo* part_info)
{
    alien_debug([&]{
      cout() << "MCGInternalLinearSolver::_solve with x0"
             << " A:" << &A << " b:" << &b << " x0:" << &x0 << " x:" << &x;
  });

  Integer error = -1;

  m_system_timer.start();
  if (_matrixChanged(A)) {
    delete m_system;
    _registerKey(A, b, x0, x);
    m_system = _createSystem(A, b, x0, x, part_info);
    if (A.m_elliptic_split_tag) {
      m_system->setEquationType(A.m_equation_type);
    }
  } else {
    if (_rhsChanged(b)) {
      m_b_key = b.m_key;
      m_system->setRhs(&b.m_bvector);
    }

    if (_x0Changed(x0)) {
      m_x0_key = x0.m_key;
      m_system->setInitSol(&x0.m_bvector);
    }
  }

  m_solver->setMatrixUpdate(m_A_update);
  m_solver->setRhsUpdate(m_b_update);
  m_solver->setx0Update(m_x0_update);

  m_system_timer.stop();

  m_solve_timer.start();
  error = m_solver->solve(m_system->getImpl(), &m_mcg_status);
  m_solve_timer.stop();

  return error;
}

MCGInternalLinearSolver::MCGSolverLinearSystem*
MCGInternalLinearSolver::_createSystem(const MCGMatrixType& A, const MCGVectorType& b,
    MCGVectorType& x, MCGSolver::PartitionInfo* part_info)
{
  return MCGSolverLinearSystem::create(
      A.m_matrix[0][0].get(), &b.m_bvector, &x.m_bvector, part_info, m_mpi_info);
}

MCGInternalLinearSolver::MCGSolverLinearSystem*
MCGInternalLinearSolver::_createSystem(const MCGMatrixType& A, const MCGVectorType& b,
    const MCGVectorType& x0, MCGVectorType& x, MCGSolver::PartitionInfo* part_info)
{
  return MCGSolverLinearSystem::create(A.m_matrix[0][0].get(), &b.m_bvector, &x.m_bvector,
      &x0.m_bvector, part_info, m_mpi_info);
}

/*---------------------------------------------------------------------------*/

const Alien::SolverStatus&
MCGInternalLinearSolver::getStatus() const
{
  return m_status;
}

void
MCGInternalLinearSolver::printInfo() const
{
  double total_solve_time = m_solve_timer.getElapse() + m_system_timer.getElapse();
  alien_info([&] {
    cout() << "\n|----------------------------------------------------|\n"
                "| Linear Solver        : MCGSolver " << m_version << "\n"
                "|----------------------------------------------------|\n"
           << std::scientific << std::setprecision(4)
           << "| total num of iter           : " << m_total_iter_num << "\n"
           << "| solve num                   : " << m_solve_num << "\n"
           << "| init time                   : " << m_init_timer.getElapse() << "\n"
           << "| prepare time                : " << m_prepare_timer.getElapse() << "\n"
           << "| total solver time           : " << total_solve_time << "\n"
           << "| |--system time              : " << m_system_timer.getElapse() << " "
           << m_system_timer.getElapse() / total_solve_time << "\n"
           << "| |--solve time               : " << m_solve_timer.getElapse() << " "
           << m_solve_timer.getElapse() / total_solve_time << "\n"
           << "|    |--internal allocate time: " << m_int_total_allocate_time << " "
           << m_int_total_allocate_time / total_solve_time << "\n"
           << "|    |--internal init time    : " << m_int_total_init_time << " "
           << m_int_total_init_time / total_solve_time << "\n"
           << "|    |--internal udpdate time : " << m_int_total_update_time << " "
           << m_int_total_update_time / total_solve_time << "\n"
           << "|    |--internal solve time   : " << m_int_total_solve_time << " "
           << m_int_total_solve_time / total_solve_time << "\n"
           << std::defaultfloat
           << "|----------------------------------------------------|\n";
  });
}

bool
MCGInternalLinearSolver::solve(IMatrix const& A, IVector const& b, IVector& x)
{
#if 0
  if(m_logger)
    m_logger->start(eStep::solve);
#endif
  m_prepare_timer.start();
  Integer error = -1;

  if (m_parallel_mng == nullptr)
    return true;

  alien_debug([&] {
   cout() << "MCGInternalLinearSolver::solve A timestamp: " << A.impl()->timestamp();
   cout() << "MCGInternalLinearSolver::solve b timestamp: " << b.impl()->timestamp();
   cout() << "MCGInternalLinearSolver::solve x timestamp: " << x.impl()->timestamp();
  });

  using namespace Alien;
  using namespace Alien::MCGInternal;

  MCGSolver::PartitionInfo* part_info = nullptr;

  if (A.impl()->hasFeature("composite")) {
    throw Alien::FatalErrorException("composite no more supported with MCGSolver");
  }
  else {
    m_A_update = A.impl()->timestamp() > m_A_time_stamp;
    m_A_update = b.impl()->timestamp() > m_b_time_stamp;
    m_A_time_stamp = A.impl()->timestamp();
    m_b_time_stamp = b.impl()->timestamp();

    const MCGMatrix& matrix = A.impl()->get<BackEnd::tag::mcgsolver>();

    const MCGVector& rhs = b.impl()->get<BackEnd::tag::mcgsolver>();
    MCGVector& sol = x.impl()->get<BackEnd::tag::mcgsolver>(true);

    if (m_use_mpi) {
      ConstArrayView<int> offsets;
      UniqueArray<Integer> blockOffsets;
      int block_size;

      if (A.impl()->block()) {
        computeBlockOffsets(matrix.distribution(), *A.impl()->block(), blockOffsets);
        int blockSize = A.impl()->block()->size();
#ifdef ALIEN_USE_ARCANE
        offsets = blockOffsets.constView();
#else
        offsets = ConstArrayView<int>(blockOffsets);
#endif
        block_size = blockSize;
        m_part_info = new MCGSolver::PartitionInfo;
        m_part_info->init((int*)offsets.data(), offsets.size(), block_size);
      }
      else {
        Integer loffset = matrix.distribution().rowOffset();
        Integer nproc = m_parallel_mng->commSize();
        UniqueArray<Integer> scalarOffsets;
        scalarOffsets.resize(nproc + 1);

        mpAllGather(m_parallel_mng, ConstArrayView<int>(1, &loffset),
            ArrayView<int>(nproc, dataPtr(scalarOffsets)));

        scalarOffsets[nproc] = matrix.distribution().globalRowSize();
#ifdef ALIEN_USE_ARCANE
        offsets = scalarOffsets.constView();
#else
        offsets = ConstArrayView<int>(scalarOffsets);
#endif
        block_size = 1;
        m_part_info = new MCGSolver::PartitionInfo;
        m_part_info->init(offsets.data(), offsets.size(), block_size);
      }
    }

    m_prepare_timer.stop();
    try {
      error = _solve(*matrix.internal(), *rhs.internal(), *sol.internal(), m_part_info);
    } catch (...) {
      // all MCGSolver exceptions are unrecoverable
      exit(EXIT_FAILURE);
    }
  }

  m_status.residual = m_mcg_status.m_residual;
  m_status.iteration_count = m_mcg_status.m_num_iter;
  m_solve_num += 1;
  m_total_iter_num += m_mcg_status.m_num_iter;

  m_int_total_solve_time += m_mcg_status.m_solve_time;
  m_int_total_allocate_time += m_mcg_status.m_allocate_time;
  m_int_total_init_time += m_mcg_status.m_init_time;
  m_int_total_update_time += m_mcg_status.m_update_time;

  if (error == 0) {
    m_status.succeeded = true;
    m_status.error = 0;

    printInfo();
    alien_info([&] {
      cout() << "Resolution info      :";
      cout() << "Resolution status    : OK";
      cout() << "Residual             : " << m_mcg_status.m_residual;
      cout() << "Number of iterations : " << m_mcg_status.m_num_iter;
    });
#if 0
    if(m_logger)
      m_logger->stop(eStep::solve, m_status);
#endif
    return true;
  } else {
    m_status.succeeded = false;
    m_status.error = m_mcg_status.m_error;
    if (m_output_level > 0 && m_parallel_mng->commRank() == 0) {
      printInfo();

      alien_info([&] {
        cout() << "Resolution status      : Error";
        cout() << "Error code             : " << m_mcg_status.m_error;
      });
    }
#if 0    
    if(m_logger)
      m_logger->stop(eStep::solve, m_status);
#endif
    return false;
  }
}

void
MCGInternalLinearSolver::setEdgeWeight(const IMatrix& E)
{
  const MCGMatrix& ew_matrix = E.impl()->get<BackEnd::tag::mcgsolver>();

  const auto* edge_weightp = ew_matrix.internal()->m_matrix[0][0]->getVal();
  const auto n_edge = ew_matrix.internal()->m_matrix[0][0]->getProfile().getNElems();

  m_edge_weight.resize(n_edge);
  std::copy(edge_weightp, edge_weightp + n_edge, m_edge_weight.begin());
}

bool
MCGInternalLinearSolver::_systemChanged(const MCGInternalLinearSolver::MCGMatrixType& A,
    const MCGInternalLinearSolver::MCGVectorType& b,
    const MCGInternalLinearSolver::MCGVectorType& x)
{
  if (m_system == nullptr) {
    return true;
  }

  if (A.m_matrix[0][0].get() != m_system->getMatrix()) {
    return true;
  }
  if (m_A_key != A.m_key) {
    return true;
  }

  if (&b.m_bvector != m_system->getRhs()) {
    return true;
  }
  if (m_b_key != b.m_key) {
    return true;
  }

  if (&x.m_bvector != m_system->getSol()) {
    return true;
  }

  if (m_x_key != x.m_key) {
    return true;
  }

  return false;
}

bool
MCGInternalLinearSolver::_matrixChanged(const MCGInternalLinearSolver::MCGMatrixType& A)
{
  if (m_system == nullptr) {
    return true;
  }

  if (A.m_matrix[0][0].get() != m_system->getMatrix()) {
    return true;
  }
  if (m_A_key != A.m_key) {
    return true;
  }

  return false;
}

bool
MCGInternalLinearSolver::_rhsChanged(const MCGVectorType& b)
{
  if (&b.m_bvector != m_system->getRhs()) {
    return true;
  }
  if (m_b_key != b.m_key) {
    return true;
  }

  return false;
}

bool
MCGInternalLinearSolver::_x0Changed(const MCGVectorType& x0)
{
  if (&x0.m_bvector != m_system->getInitSol()) {
    return true;
  }
  if (m_x0_key != x0.m_key) {
    return true;
  }

  return false;
}

bool
MCGInternalLinearSolver::_systemChanged(const MCGInternalLinearSolver::MCGMatrixType& A,
    const MCGInternalLinearSolver::MCGVectorType& b,
    const MCGInternalLinearSolver::MCGVectorType& x0,
    const MCGInternalLinearSolver::MCGVectorType& x)
{
  if (_systemChanged(A, b, x)) {
    return true;
  }

  if (&x0.m_bvector != m_system->getInitSol()) {
    return true;
  }
  if (m_x0_key != x0.m_key) {
    return true;
  }
  return false;
}

void
MCGInternalLinearSolver::_registerKey(const MCGInternalLinearSolver::MCGMatrixType& A,
    const MCGInternalLinearSolver::MCGVectorType& b,
    const MCGInternalLinearSolver::MCGVectorType& x)
{
  m_A_key = A.m_key;
  m_b_key = b.m_key;
  m_x_key = x.m_key;
}

void
MCGInternalLinearSolver::_registerKey(const MCGInternalLinearSolver::MCGMatrixType& A,
    const MCGInternalLinearSolver::MCGVectorType& b,
    const MCGInternalLinearSolver::MCGVectorType& x0,
    const MCGInternalLinearSolver::MCGVectorType& x)
{
  _registerKey(A, b, x);
  m_x0_key = m_x0_key;
}

ILinearSolver*
MCGInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsMCGSolver* options)
{
  return new MCGInternalLinearSolver(p_mng, options);
}
} // namespace Alien
