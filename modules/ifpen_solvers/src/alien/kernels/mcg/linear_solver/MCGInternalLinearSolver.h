/*
 * MCGInternalLinearSolver.h
 *
 *  Created on: 22 dec. 2014
 *      Author: gratienj
 */
#ifndef ALIEN_KERNELS_MCG_LINEARSOLVER_MCGINTERNALLINEARSOLVER_H
#define ALIEN_KERNELS_MCG_LINEARSOLVER_MCGINTERNALLINEARSOLVER_H

#include <memory>
#include <chrono>

#include <MCGSolver/ILinearSystem.h>
#include <Common/Utils/Machine/MachineInfo.h>
#include <Common/Utils/ParallelEnv.h>
#include <Graph/OrderingType.h>
#include <Precond/PrecondOptionsEnum.h>
#include <Solvers/AMG/AMGProperty.h>
#include <Solvers/SolverProperty.h>
#include <MCGSolver/Status.h>
#include <MCGSolver/SolverOptionsEnum.h>
#include <MCGSolver/ILinearSolver.h>

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

class SolverStater;

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

    const double getElapse() const { return m_elapse.count(); }
  };

  typedef SolverStatus Status;
  typedef MCGMatrix MatrixType;
  typedef MCGVector VectorType;

  typedef MCGInternal::MatrixInternal MCGMatrixType;
  typedef MCGInternal::VectorInternal MCGVectorType;

  typedef SimpleCSRMatrix<Real> CSRMatrixType;
  typedef SimpleCSRVector<Real> CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real> CSRInternalMatrixType;

 public:
  MCGInternalLinearSolver() = delete;
  MCGInternalLinearSolver(const MCGInternalLinearSolver&) = delete;

  /** Constructeur de la classe */
  MCGInternalLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsMCGSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~MCGInternalLinearSolver();

 public:
  //! Initialisation
  // void init(int argv,char const** argc);
  void init();
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);
  void updateParameters();

  // void setDiagScal(double* Diag, int size);
  //! Finalize
  void end();

  String getBackEndName() const { return "mcgsolver"; }

  //! Rï¿œsolution du systï¿œme linï¿œaire
  bool solve(IMatrix const& A, IVector const& b, IVector& x);

  //! Indicateur de support de r�solution parall�le
  bool hasParallelSupport() const { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const;

  const SolverStat& getSolverStat() const { return m_stat; }

  String getName() const { return "mcgsolver"; }

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag)
  {
    alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
  }

  virtual void setEdgeWeight(const IMatrix& E) final;

  void printInfo() const;

 private:
  Integer _solve(const MCGMatrixType& A, const MCGVectorType& b, MCGVectorType& x,
      MCGSolver::PartitionInfo* part_info = nullptr);
  Integer _solve(const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x0,
      MCGVectorType& x, MCGSolver::PartitionInfo* part_info = nullptr);

  bool _systemChanged(
      const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x);
  bool _systemChanged(const MCGMatrixType& A, const MCGVectorType& b,
      const MCGVectorType& x0, const MCGVectorType& x);
  bool _matrixChanged(const MCGMatrixType& A);
  bool _rhsChanged(const MCGVectorType& b);
  bool _x0Changed(const MCGVectorType& x0);

  // bool _systemChanged(const MCGMatrixType& A,const MCGVectorType& b,const
  // MCGVectorType& x0,const MCGVectorType& x);
  void _registerKey(
      const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x);
  void _registerKey(const MCGMatrixType& A, const MCGVectorType& b,
      const MCGVectorType& x0, const MCGVectorType& x);

  typedef MCGSolver::ILinearSystem<double, MCGSolver::LinearSystem<double>>
      MCGSolverLinearSystem;

  MCGSolverLinearSystem* _createSystem(const MCGMatrixType& A, const MCGVectorType& b,
      MCGVectorType& x, MCGSolver::PartitionInfo* part_info = nullptr);

  MCGSolverLinearSystem* _createSystem(const MCGMatrixType& A, const MCGVectorType& b,
      const MCGVectorType& x0, MCGVectorType& x,
      MCGSolver::PartitionInfo* part_info = nullptr);

 protected:
  MCGSolver::ILinearSolver<MCGSolver::LinearSolver>* m_solver = nullptr;

 private:
  //! Structure interne du solveur
  std::string m_version;
  bool m_use_mpi = false;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  MCGSolver::MachineInfo* m_machine_info = nullptr; // TODO: use shared_ptr
  mpi::MPIInfo* m_mpi_info = nullptr; // TODO: use shared_ptr
  MCGSolver::PartitionInfo* m_part_info = nullptr; // TODO: use shared_ptr

  MCGSolver::Status m_mcg_status;
  Alien::SolverStatus m_status;

  //! Preconditioner options
  MCGSolver::ePrecondType m_precond_opt = MCGSolver::PrecNone;

  //! Solver parameters
  MCGSolver::eKrylovType m_solver_opt = MCGSolver::BiCGS;
  Integer m_max_iteration = 1000;
  Real m_precision = 1e-6;

  String m_matrix_file_name;

  // multithread options
  bool m_use_thread = false;
  Integer m_num_thread = 0;

  int m_current_ctx_id = 0;

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

  MCGSolverLinearSystem* m_system = nullptr;

  MCGInternal::UniqueKey m_A_key;
  Integer m_A_time_stamp = 0;
  bool m_A_update = true;

  MCGInternal::UniqueKey m_b_key;
  Integer m_b_time_stamp = 0;
  bool m_b_update = true;

  MCGInternal::UniqueKey m_x_key;

  MCGInternal::UniqueKey m_x0_key;
  Integer m_x0_time_stamp = 0;
  bool m_x0_update = true;

  std::vector<int> m_edge_weight;
#if 0  
  std::unique_ptr<ILogger> m_logger;
#endif
};

} // namespace Alien
#endif /* ALIEN_KERNELS_MCG_LINEARSOLVER_MCGINTERNALLINEARSOLVER_H */