/*
 * HTSInternalLinearSolver.h
 *
 *  Created on: 22 déc. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_KERNELS_HTS_LINEARSOLVER_HTSINTERNALLINEARSOLVER_H
#define ALIEN_KERNELS_HTS_LINEARSOLVER_HTSINTERNALLINEARSOLVER_H

#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Core/Backend/IInternalLinearSolverT.h>
#include <ALIEN/Kernels/HTS/LinearSolver/HTSOptionTypes.h>
#include <ALIEN/Expression/Solver/SolverStats/SolverStater.h>
#include <ALIEN/Core/Backend/IInternalLinearSolverT.h>
#include <ALIEN/Utils/Trace/ObjectWithTrace.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRMatrix.h>
#include <ALIEN/Alien-IFPENSolversPrecomp.h>

class IOptionsHTSSolver;

namespace Alien {

class SolverStater;

class ALIEN_IFPENSOLVERS_EXPORT HTSInternalLinearSolver
//: public IInternalLinearSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >
: public ILinearSolver,
  public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;

  typedef SimpleCSRMatrix<Arccore::Real>      CSRMatrixType;
  typedef SimpleCSRVector<Arccore::Real>      CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Arccore::Real>  CSRInternalMatrixType;

 public:
  /** Constructeur de la classe */
  HTSInternalLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr, IOptionsHTSSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~HTSInternalLinearSolver(){}

 public:
  //! Initialisation
  void init(int argv,char const** argc);
  void init();
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);
  void updateParameters() ;

  //void setDiagScal(double* Diag, int size);
  //! Finalize
  void end();

  Arccore::String getBackEndName() const {
    return "htssolver" ;
  }

  bool solve(IMatrix const& A, IVector const& b, IVector& x);
  bool solve(const CSRMatrixType& A, const CSRVectorType& b, CSRVectorType& x);

  bool solve() ;

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const
  {
    return true ;
  }

  std::shared_ptr<ILinearAlgebra> algebra() const ;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const ;
  Alien::SolverStatus& getStatusRef() {
    return m_status ;
  }

  const SolverStat & getSolverStat() const { return m_stater; }
  SolverStater& getSolverStater() { return m_stater; }

  Arccore::String getName() const { return "htssolver"; }


  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag) {
    alien_warning([&] {
        cout() << "Null Space Constant Option not yet implemented" ;
      });
  }

  void printInfo() const ;
  void printInfo() ;
  void printCurrentTimeInfo() {}

  struct RunOp
  {
    typedef HartsSolver::CSRMatrix<Arccore::Real,1> MatrixType ;
    typedef MatrixType::VectorDataType     VectorDataType ;
    HartsSolver::HTSSolver*         m_solver;
    MatrixType&                     m_A;
    VectorDataType const*           m_b;
    VectorDataType*                 m_x;
    HartsSolver::HTSSolver::Status& m_status ;
    HartsSolver::HTSSolver::ContextType& m_context ;

    RunOp(HartsSolver::HTSSolver* solver,
          HartsSolver::CSRMatrix<Arccore::Real,1>& A,
          const double* b,
          double* x,
          HartsSolver::HTSSolver::Status& status,
          HartsSolver::HTSSolver::ContextType& context)
    : m_solver(solver)
    , m_A(A)
    , m_b(b)
    , m_x(x)
    , m_status(status)
    , m_context(context)
    {}

    template<bool is_mpi, bool use_simd, HARTS::eThreadEnvType th_env>
    void run()
    {
      m_solver->solve<is_mpi,use_simd,th_env>(m_A,m_b,m_x,m_status,m_context) ;
    }
  };

 private:

  void updateLinearSystem();
  inline void _startPerfCount();
  inline void _endPerfCount();

 protected :

  std::unique_ptr<HartsSolver::HTSSolver> m_hts_solver ;
 private:
  //! Structure interne du solveur

  bool                         m_use_mpi = false ;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  RunTimeSystem::MachineInfo   m_machine_info ;
  HARTS::Runtime::Configuration  m_runtime_configuration ;
  //HartsSolver::MPIInfo*        m_mpi_info     = nullptr;

  //Status m_status;
  HartsSolver::HTSSolver::Status          m_hts_status;
  Alien::SolverStatus                     m_status;

  //!Preconditioner options
  //HTSOptionTypes::ePreconditioner m_precond_opt ;

  //!Solver parameters
  Arccore::Integer m_max_iteration = 0 ;
  Arccore::Real    m_precision     = 0. ;

  //!Linear system builder options
  //!@{
  //bool m_use_unit_diag ;
  //bool m_keep_diag_opt ;
  //int  m_normalize_opt ;
  //!@}


  // multithread options
  bool    m_use_thread = false;
  Arccore::Integer m_num_thread = 1 ;
  HARTS::eThreadEnvType m_thread_env_type = HARTS::PTh ;

  int m_current_ctx_id = -1 ;

  Arccore::Integer m_output_level = 0 ;

  Arccore::Integer m_solve_num          = 0 ;
  Arccore::Integer m_total_iter_num     = 0 ;
  Arccore::Real m_current_solve_time    = 0. ;
  Arccore::Real m_total_solve_time      = 0. ;
  Arccore::Real m_current_system_time   = 0. ;
  Arccore::Real m_total_system_time     = 0. ;
  Arccore::Real m_int_total_solve_time  = 0. ;
  Arccore::Real m_int_total_setup_time  = 0. ;
  Arccore::Real m_int_total_finish_time = 0. ;

  SolverStater m_stater;

  IOptionsHTSSolver* m_options = nullptr ;
  std::vector<double>  m_pressure_diag; 

};

} // namespace Alien

#endif /* HTSLINEARSOLVER_H_ */
