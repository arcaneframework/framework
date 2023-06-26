/*
 * TrilinosInternalLinearSolver.h
 *
 *  Created on: 22 déc. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_KERNELS_TRILINOS_LINEARSOLVER_TRILINOSINTERNALLINEARSOLVER_H
#define ALIEN_KERNELS_TRILINOS_LINEARSOLVER_TRILINOSINTERNALLINEARSOLVER_H

#include <alien/utils/Precomp.h>
#include <alien/AlienTrilinosPrecomp.h>
#include <alien/kernels/trilinos/linear_solver/TrilinosOptionTypes.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>

class IOptionsTrilinosSolver;

namespace Alien {

class SolverStat;

template <typename TagT>
class ALIEN_TRILINOS_EXPORT TrilinosInternalLinearSolver
    : public IInternalLinearSolver<TrilinosMatrix<Arccore::Real, TagT>,
          TrilinosVector<Arccore::Real, TagT>>,
      public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;

  // typedef TrilinosInternal::MatrixInternal<Real>  InternalMatrixType;
  typedef TrilinosMatrix<Arccore::Real, TagT> CSRMatrixType;
  typedef TrilinosVector<Arccore::Real, TagT> CSRVectorType;

 public:
  /** Constructeur de la classe */
  TrilinosInternalLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsTrilinosSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~TrilinosInternalLinearSolver() {}

 public:
  //! Initialisation
  void init(int argv, char const** argc);

  void init();

  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);

  void updateParameters();

  // void setDiagScal(double* Diag, int size);
  //! Finalize
  void end();

  String getBackEndName() const;

  typedef typename TrilinosInternal::SolverInternal<TagT> solver_type;

  // bool solve(IMatrix const& A, IVector const& b, IVector& x);
  bool solve(const CSRMatrixType& matrix, const CSRVectorType& b, CSRVectorType& x);

  // bool solve() ;

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const;

  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

  String getName() const { return "trilinos"; }

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag)
  {
    alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
  }

  void internalPrintInfo() const;
  void printInfo() const { internalPrintInfo(); }
  void printInfo() { internalPrintInfo(); }
  void printCurrentTimeInfo() {}

 private:
  void updateLinearSystem();
  inline void _startPerfCount();
  inline void _endPerfCount();

 protected:
 private:
  //! Structure interne du solveur
  bool m_initialized  = false ;
  bool m_use_mpi = false;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;

  Alien::SolverStatus m_status;

  //! Preconditioner options
  std::string m_precond_name = "ILUT";

  std::string m_solver_name = "GMRES";

  std::unique_ptr<solver_type> m_trilinos_solver;

  //! Solver parameters
  Integer m_max_iteration = 0;
  Real m_precision = 0.;

  // multithread options
  bool m_use_thread = false;
  Integer m_num_thread = 1;

  int m_current_ctx_id = -1;

  Integer m_output_level = 0;

  Integer m_solve_num = 0;
  Integer m_total_iter_num = 0;

  SolverStat m_stat;

  IOptionsTrilinosSolver* m_options = nullptr;
  std::vector<double> m_pressure_diag;
};

} // namespace Alien

#endif /* TRILINOSLINEARSOLVER_H_ */
