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
#include <ALIEN/Kernels/MCG/LinearSolver/GPUInternal.h>
#include <ALIEN/Kernels/MCG/DataStructure/MCGVector.h>
#include <ALIEN/Kernels/MCG/DataStructure/MCGMatrix.h>
#include <ALIEN/Kernels/MCG/DataStructure/MCGInternal.h>
#include <ALIEN/Kernels/MCG/LinearSolver/GPUOptionTypes.h>
#include <alien/expression/solver/solver_stats/SolverStater.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/trace/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRMatrix.h>
#include <ALIEN/Alien-IFPENSolversPrecomp.h>

class IOptionsGPUSolver;

namespace Alien {

class SolverStater;

class ALIEN_IFPENSOLVERS_EXPORT MCGInternalLinearSolver : public ILinearSolver,
                                                          public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;
  typedef MCGMatrix MatrixType;
  typedef MCGVector VectorType;

  typedef MCGInternal::MatrixInternal  MCGMatrixType;
  typedef MCGInternal::VectorInternal  MCGVectorType;
  typedef SimpleCSRMatrix<Real>      CSRMatrixType;
  typedef SimpleCSRVector<Real>      CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real>  CSRInternalMatrixType;

 public:
  /** Constructeur de la classe */
  MCGInternalLinearSolver(IParallelMng* parallel_mng = nullptr, IOptionsGPUSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~MCGInternalLinearSolver();

 public:
  //! Initialisation
  //void init(int argv,char const** argc);
  void init();
  void updateParallelMng(IParallelMng* pm);
  void updateParameters() ;

  //void setDiagScal(double* Diag, int size);
  //! Finalize
  void end();

  String getBackEndName() const {
    return "mcgsolver" ;
  }

  //! Rï¿œsolution du systï¿œme linï¿œaire
  bool solve(IMatrix const& A, IVector const& b, IVector& x);

  //! Indicateur de support de r�solution parall�le
  bool hasParallelSupport() const
  {
    return true ;
  }

  std::shared_ptr<ILinearAlgebra> algebra() const ;

  //! Etat du solveur
  const Alien::SolverStatus & getStatus() const ;

  const SolverStat & getSolverStat() const { return m_stat; }

  String getName() const { return "gpusolver"; }

  //! R�solution du syst�me lin�aire

  bool solve() ;


  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag) {
    alien_warning([&] {
        cout() << "Null Space Constant Option not yet implemented" ;
      });
  }

  void printInfo() const ;
  void printInfo() ;
  void printCurrentTimeInfo() {}

 private:

  bool _solve(MCGMatrixType const& A, MCGVectorType const& b, MCGVectorType& x) ;
  bool _solve(MCGMatrixType const& A, MCGVectorType const& b, MCGVectorType& x, MCGVectorType& x0) ;

  bool _solve(MCGMatrixType const& A,MCGVectorType const& diag_scal,MCGVectorType const& b, MCGVectorType& x);
  bool _solve(MCGMatrixType const& A,MCGVectorType const& diag_scal,MCGVectorType const& b, MCGVectorType& x, MCGVectorType& x0);

  bool _solveMC(CSRMatrixType const& A, CSRVectorType const& b, CSRVectorType& x) ;
  bool _solve(CSRMatrixType const& A, CSRVectorType const& b, CSRVectorType& x) ;
  void updateLinearSystem();
  inline void _startPerfCount();
  inline void _endPerfCount();

  template<int N>
  GPUSolver::System* _createSystem(MCGMatrixType const& A,
                                   MCGVectorType const& b,
                                   MCGVectorType& x) ;
  template<int N>
  GPUSolver::System* _createSystem(MCGMatrixType const& A,
                                   MCGVectorType const& diag_scal,
                                   MCGVectorType const& b,
                                   MCGVectorType& x) ;

  template<int N>
  GPUSolver::System* _createSystem(MCGMatrixType const& A,
                                   MCGVectorType const& b,
                                   MCGVectorType& x,
                                   MCGVectorType& x0) ;
  template<int N>
  GPUSolver::System* _createSystem(MCGMatrixType const& A,
                                   MCGVectorType const& diag_scal,
                                   MCGVectorType const& b,
                                   MCGVectorType& x,
                                   MCGVectorType& x0) ;

  GPUSolver::System* _computeGPUSystem(MCGMatrixType const& A,
                                       MCGVectorType const& b,
                                       MCGVectorType& x,
                                       Integer equations_num) ;

  GPUSolver::System* _computeGPUSystem(MCGMatrixType const& A,
                                       MCGVectorType const& b,
                                       MCGVectorType& x,
                                       MCGVectorType& x0,
                                       Integer equations_num) ;

  GPUSolver::System* _computeGPUSystem(MCGMatrixType const& A,
                                       MCGVectorType const& diag_scal,
                                       MCGVectorType const& b,
                                       MCGVectorType& x,
                                       Integer equations_num) ;

 GPUSolver::System* _computeGPUSystem(MCGMatrixType const& A,
                                       MCGVectorType const& diag_scal,
                                       MCGVectorType const& b,
                                       MCGVectorType& x,
                                       MCGVectorType& x0,
                                       Integer equations_num) ;
 protected :

  GPUSolver* m_gpu_solver ;
 private:
  //! Structure interne du solveur

  bool m_use_mpi ;
  IParallelMng*                m_parallel_mng ;
  MCGSolver::MachineInfo       m_machine_info ;
  MCGSolver::MPIInfo*           m_mpi_info;
  MCGSolver::ParallelContext<MCGSolver::PartitionInfo>*  m_parallel_context ;

  //Status m_status;
  GPUSolver::Status m_mcgs_status;
  Alien::SolverStatus m_status;

  //!Preconditioner options
  GPUOptionTypes::ePreconditioner m_precond_opt ;

  //!Solver parameters
  GPUOptionTypes::eSolver m_solver_opt;
  Integer m_max_iteration ;
  Real m_precision ;

  //!Linear system builder options
  //!@{
  bool m_use_unit_diag ;
  bool m_keep_diag_opt ;
  Integer  m_normalize_opt ;
  //!@}

  String m_matrix_file_name ;

  // multithread options
  bool m_use_thread;
  Integer m_num_thread;

  int m_current_ctx_id ;

  Integer m_output_level ;

  Integer m_solve_num;
  Integer m_total_iter_num;
  Real m_current_solve_time;
  Real m_total_solve_time;
  Real m_current_system_time;
  Real m_total_system_time;
  Real m_int_total_solve_time;
  Real m_int_total_setup_time;
  Real m_int_total_finish_time;

  SolverStat   m_stat ;
  SolverStater m_stater;

  IOptionsGPUSolver* m_options;
  std::vector<double>  m_pressure_diag; 

  std::string m_dir;
  std::map<std::string,int> m_dir_enum;

};

} // namespace Alien

#endif /* PETSCLINEARSOLVER_H_ */
