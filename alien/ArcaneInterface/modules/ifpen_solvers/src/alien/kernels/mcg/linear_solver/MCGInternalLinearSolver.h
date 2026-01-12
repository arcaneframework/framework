// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <chrono>

#include <Common/index.h>
#include <MCGSolver/MCGSolver.h>
#include <Graph/OrderingType.h>
#include <Precond/PrecondOptionsEnum.h>
#include <Solvers/AMG/AMGProperty.h>
#include <Solvers/SolverProperty.h>
#include <MCGSolver/SolverOptionsEnum.h>

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include "alien/AlienIFPENSolversPrecomp.h"
#include "alien/kernels/mcg/data_structure/MCGVector.h"
#include "alien/kernels/mcg/data_structure/MCGMatrix.h"
#include "alien/kernels/mcg/data_structure/MCGInternal.h"
#include "alien/kernels/mcg/linear_solver/MCGOptionTypes.h"

class IOptionsMCGSolver;

namespace Alien {

class SolverStat;

class ALIEN_IFPEN_SOLVERS_EXPORT MCGInternalLinearSolver : public ILinearSolver,
                                                           public ObjectWithTrace
{
 private:
  class AlienKOpt2MCGKOpt : public ObjectWithTrace
  {
   public:
    typedef std::tuple<MCGOptionTypes::eKernelType, bool, bool> KConfigType;

    AlienKOpt2MCGKOpt(const AlienKOpt2MCGKOpt&) = delete;

    static MCGSolver::eKernelType getKernelOption(const KConfigType& kernel_type);

   private:
    static std::unique_ptr<AlienKOpt2MCGKOpt> m_instance;

    std::map<KConfigType, MCGSolver::eKernelType> m_option_translate;

    AlienKOpt2MCGKOpt();
  };

  class Timer
  {
   private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
    std::chrono::time_point<std::chrono::system_clock> m_end;
    std::chrono::duration<double> m_elapse;

   public:
    Timer()
    : m_elapse(0)
    {}

    inline void start() { m_start = std::chrono::system_clock::now(); }

    inline void stop()
    {
      m_end = std::chrono::system_clock::now();
      m_elapse += m_end - m_start;
    }

    [[nodiscard]] double getElapse() const { return m_elapse.count(); }
  };

  typedef SolverStatus Status;

  typedef MCGInternal::MatrixInternal<double,MCGInternal::eMemoryDomain::Host> MCGMatrixType;
  typedef MCGInternal::VectorInternal<double,MCGInternal::eMemoryDomain::Host> MCGVectorType;
  typedef MCGInternal::MatrixInternal<double,MCGInternal::eMemoryDomain::Device> MCGDeviceMatrixType;
  typedef MCGInternal::VectorInternal<double,MCGInternal::eMemoryDomain::Device> MCGDeviceVectorType;


  typedef SimpleCSRMatrix<Real> CSRMatrixType;
  typedef SimpleCSRVector<Real> CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real> CSRInternalMatrixType;

 public:
  MCGInternalLinearSolver() = delete;
  MCGInternalLinearSolver(const MCGInternalLinearSolver&) = delete;

  /** Constructeur de la classe */
  explicit MCGInternalLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsMCGSolver* options = nullptr);

  /** Destructeur de la classe */
  ~MCGInternalLinearSolver() override;

 public:
  //! Initialisation
  void init() override;
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm) override;
  //void updateParameters();

  // void setDiagScal(double* Diag, int size);
  //! Finalize
  void end() override;

  String getBackEndName() const override { return "mcgsolver"; }

  //! Linear system solve
  bool solve(IMatrix const& A, IVector const& b, IVector& x) override;

  //! Indicateur de support de r�solution parall�le
  bool hasParallelSupport() const override { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const override;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const override;

  const SolverStat& getSolverStat() const override { return m_stat; }

  String getName() const { return "mcgsolver"; }

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag) override
  {
    alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
  }

  void printInfo() const;

  void startNonLinear() final;

 private:

  Integer _solve(const MCGMatrixType& A, const MCGVectorType& b, MCGVectorType& x,
      const std::shared_ptr<const MCGSolver::PartitionInfo<int32_t>>& part_info);
  Integer _solve(const MCGDeviceMatrixType& A, const MCGDeviceVectorType& b, MCGDeviceVectorType& x,
      const std::shared_ptr<const MCGSolver::PartitionInfo<int32_t>>& part_info);

  Integer _solve(const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x0,
      MCGVectorType& x,const std::shared_ptr<const MCGSolver::PartitionInfo<int32_t>>& part_info);
  Integer _solve(const MCGDeviceMatrixType& A, const MCGDeviceVectorType& b, const MCGDeviceVectorType& x0,
    MCGDeviceVectorType& x,const std::shared_ptr<const MCGSolver::PartitionInfo<int32_t>>& part_info);

  bool _systemChanged(const MCGMatrixType& A, const MCGVectorType& b) const;
  bool _systemChanged(const MCGDeviceMatrixType& A, const MCGDeviceVectorType& b) const;

  bool _systemChanged(const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x0) const;
  bool _systemChanged(const MCGDeviceMatrixType& A, const MCGDeviceVectorType& b, const MCGDeviceVectorType& x0) const;

  bool _matrixChanged(const MCGMatrixType& A) const;
  bool _matrixChanged(const MCGDeviceMatrixType& A) const;

  bool _rhsChanged(const MCGVectorType& b) const;
  bool _rhsChanged(const MCGDeviceVectorType& b) const;

  bool _x0Changed(const MCGVectorType& x0) const;
  bool _x0Changed(const MCGDeviceVectorType& x0) const;

  void _registerKey(const MCGMatrixType& A, const MCGVectorType& b);
  void _registerKey(const MCGDeviceMatrixType& A, const MCGDeviceVectorType& b);

  void _registerKey(const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x0);
  void _registerKey(const MCGDeviceMatrixType& A, const MCGDeviceVectorType& b, const MCGDeviceVectorType& x0);

  bool _hostSolver(MCGSolver::eKernelType kernel);

  typedef MCGSolver::LinearSystem<double, MCGSolver::Int32SparseIndex> MCGSolverLinearSystem;
  typedef MCGSolver::GPULinearSystem<double, MCGSolver::Int32SparseIndex> MCGSolverDeviceLinearSystem;

 protected:
  std::unique_ptr<MCGSolver::LinearSolver> m_solver;

 private:
  //! Structure interne du solveur
  std::string m_version;
  bool m_use_mpi = false;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  MCGSolver::MachineInfo* m_machine_info = nullptr; // TODO: use shared_ptr
  mpi::MPIInfo* m_mpi_info = nullptr; // TODO: use shared_ptr
  std::shared_ptr<MCGSolver::PartitionInfo<int32_t>> m_part_info;

  MCGSolver::Status m_mcg_status;
  Alien::SolverStatus m_status;

  //! Preconditioner options
  MCGSolver::ePrecondType m_precond_opt = MCGSolver::PrecNone;

  //! Solver parameters
  MCGSolver::eKernelType m_kernel = MCGSolver::eKernelType::CPU_CBLAS_BCSR;
  MCGSolver::eKrylovType m_solver_opt = MCGSolver::BiCGS;
  Integer m_max_iteration = 1000;
  Real m_precision = 1e-6;

  String m_matrix_file_name;

  // multithread options
  bool m_use_thread = false;
  Integer m_num_thread = 0;

  Integer m_output_level = 0;

  Integer m_solve_num = 0;
  Integer m_total_iter_num = 0;

  Timer m_init_timer;
  Timer m_prepare_timer;
  Timer m_system_timer;
  Timer m_solve_timer;

  // From internal MCGSolver timing
  Real m_int_total_solve_time = 0;
  Real m_int_total_allocate_time = 0;
  Real m_int_total_init_time = 0;
  Real m_int_total_update_time = 0;

  SolverStat m_stat;

  IOptionsMCGSolver* m_options = nullptr;
  std::vector<double> m_pressure_diag;

  std::string m_dir;
  std::map<std::string, int> m_dir_enum;

  std::unique_ptr<MCGSolverLinearSystem> m_system;
  std::unique_ptr<MCGSolverDeviceLinearSystem> m_device_system;

  MCGSolver::UniqueKey m_A_key;
  int64_t m_A_time_stamp = 0;
  bool m_A_update = true;

  MCGSolver::UniqueKey m_b_key;
  int64_t m_b_time_stamp = 0;
  bool m_b_update = true;

  MCGSolver::UniqueKey m_x_key;

  MCGSolver::UniqueKey m_x0_key;
  int64_t m_x0_time_stamp = 0;
  bool m_x0_update = true;

  std::vector<int> m_edge_weight;

  std::shared_ptr<MCGSolver::ILogMng> m_mcg_log;
  std::shared_ptr<MCGSolver::ILogMng> m_mcg_mpi_log;

#if 0
  std::unique_ptr<ILogger> m_logger;
#endif
};

}

