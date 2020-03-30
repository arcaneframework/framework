/*
 * MCGInternalLinearSolver.h
 *
 *  Created on: 22 dec. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_KERNELS_MCG_LINEARSOLVER_MCGINTERNALLINEARSOLVER_H
#define ALIEN_KERNELS_MCG_LINEARSOLVER_MCGINTERNALLINEARSOLVER_H

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/kernels/mcg/linear_solver/GPUInternal.h>
#include <alien/kernels/mcg/data_structure/MCGVector.h>
#include <alien/kernels/mcg/data_structure/MCGMatrix.h>
#include <alien/kernels/mcg/data_structure/MCGInternal.h>
#include <alien/kernels/mcg/linear_solver/GPUOptionTypes.h>
#include <alien/expression/solver/solver_stats/SolverStater.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/trace/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRMatrix.h>
#include <alien/AlienIFPENSolversPrecomp.h>

class IOptionsMCGSolver;

namespace Alien {

class SolverStater;

class ALIEN_EXTERNALPACKAGES_EXPORT MCGInternalLinearSolver : public ILinearSolver,
                                                              public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;
  typedef MCGMatrix MatrixType;
  typedef MCGVector VectorType;

  typedef MCGInternal::MatrixInternal MCGMatrixType;
  typedef MCGInternal::VectorInternal MCGVectorType;
  typedef MCGInternal::CompositeVectorInternal MCGCompositeVectorType;

  typedef SimpleCSRMatrix<Real> CSRMatrixType;
  typedef SimpleCSRVector<Real> CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real> CSRInternalMatrixType;

 public:
  MCGInternalLinearSolver() = delete;
  MCGInternalLinearSolver(const MCGInternalLinearSolver&) = delete;

  /** Constructeur de la classe */
  MCGInternalLinearSolver(
      IParallelMng* parallel_mng = nullptr, IOptionsMCGSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~MCGInternalLinearSolver();

 public:
  //! Initialisation
  // void init(int argv,char const** argc);
  void init();
  void updateParallelMng(IParallelMng* pm);
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

  String getName() const { return "gpusolver"; }

  //! R�solution du syst�me lin�aire

  bool solve();

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag)
  {
    alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
  }

  void printInfo() const;
  void printInfo();
  void printCurrentTimeInfo() {}

 private:
  bool _solve(const MCGMatrixType& A, const MCGVectorType& b, MCGVectorType& x,
      MCGSolver::PartitionInfo* part_info = nullptr);
  bool _solve(const MCGMatrixType& A, const MCGVectorType& b, const MCGVectorType& x0,
      MCGVectorType& x, MCGSolver::PartitionInfo* part_info = nullptr);

  bool _solve(const MCGMatrixType& A, const MCGCompositeVectorType& b,
      MCGCompositeVectorType& x, MCGSolver::PartitionInfo* part_info = nullptr);
  bool _solve(const MCGMatrixType& A, const MCGCompositeVectorType& b,
      const MCGCompositeVectorType& x0, MCGCompositeVectorType& x,
      MCGSolver::PartitionInfo* part_info = nullptr);

  void updateLinearSystem();
  inline void _startPerfCount();
  inline void _endPerfCount();

  MCGSolver::LinearSystem* _createSystem(const MCGMatrixType& A, const MCGVectorType& b,
      MCGVectorType& x, MCGSolver::PartitionInfo* part_info = nullptr);

  MCGSolver::LinearSystem* _createSystem(const MCGMatrixType& A, const MCGVectorType& b,
      const MCGVectorType& x0, MCGVectorType& x,
      MCGSolver::PartitionInfo* part_info = nullptr);

  MCGSolver::LinearSystem* _createSystem(const MCGMatrixType& A,
      const MCGCompositeVectorType& b, MCGCompositeVectorType& x,
      MCGSolver::PartitionInfo* part_info = nullptr);

  MCGSolver::LinearSystem* _createSystem(const MCGMatrixType& A,
      const MCGCompositeVectorType& b, const MCGCompositeVectorType& x0,
      MCGCompositeVectorType& x, MCGSolver::PartitionInfo* part_info = nullptr);

 protected:
  MCGSolver::LinearSolver* m_solver = nullptr;

 private:
  //! Structure interne du solveur
  bool m_use_mpi = false;
  IParallelMng* m_parallel_mng = nullptr;
  MCGSolver::MachineInfo* m_machine_info = nullptr;
  MCGSolver::MPIInfo* m_mpi_info = nullptr;

  MCGSolver::Status m_mcg_status;
  Alien::SolverStatus m_status;

  //! Preconditioner options
  MCGOptionTypes::ePreconditioner m_precond_opt = MCGOptionTypes::NonePC;

  //! Solver parameters
  MCGOptionTypes::eSolver m_solver_opt = MCGOptionTypes::BiCGStab;
  Integer m_max_iteration = 1000;
  Real m_precision = 1e-6;

  //! Linear system builder options
  //!@{
  bool m_use_unit_diag = false;
  bool m_keep_diag_opt = false;
  Integer m_normalize_opt = 0;
  //!@}

  String m_matrix_file_name;

  // multithread options
  bool m_use_thread = false;
  Integer m_num_thread = 0;

  int m_current_ctx_id = 0;

  Integer m_output_level = 0;

  Integer m_solve_num = 0;
  Integer m_total_iter_num = 0;
  Real m_current_solve_time = 0;
  Real m_total_solve_time = 0;
  Real m_current_system_time = 0;
  Real m_total_system_time = 0;
  Real m_int_total_solve_time = 0;
  Real m_int_total_setup_time = 0;
  Real m_int_total_finish_time = 0;

  SolverStat m_stat;
  SolverStater m_stater;

  IOptionsMCGSolver* m_options = nullptr;
  std::vector<double> m_pressure_diag;

  std::string m_dir;
  std::map<std::string, int> m_dir_enum;
};

} // namespace Alien

#endif /* PETSCLINEARSOLVER_H_ */
