/*
 * PETScLinearSolver.h
 *
 *  Created on: 22 déc. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_IFPLINEARSOLVER_H_
#define ALIEN_IFPLINEARSOLVER_H_

#include <alien/utils/Precomp.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/kernels/ifp/data_structure/IFPVector.h>
#include <alien/kernels/ifp/data_structure/IFPMatrix.h>
#include <alien/AlienIFPENSolversPrecomp.h>

class IOptionsIFPLinearSolver;

namespace Alien {

class SolverStat;

class ALIEN_IFPEN_SOLVERS_EXPORT IFPInternalLinearSolver : public ILinearSolver
{
 private:
  typedef SolverStatus Status;
  typedef IMatrix MatrixType;
  typedef IVector VectorType;
  typedef IVector VectorSolType;

 public:
  /** Constructeur de la classe */
  IFPInternalLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      IOptionsIFPLinearSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~IFPInternalLinearSolver();

 public:
  void init();

  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);

  //! Finalize
  void end();

  //! return package back end name
  String getBackEndName() const { return "ifpsolver"; }

  bool solve(const MatrixType& A, const VectorType& b, VectorSolType& x);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  //! Algèbre linéaire compatible
  std::shared_ptr<ILinearAlgebra> algebra() const
  {
    return std::shared_ptr<ILinearAlgebra>();
  }

  //! Etat du solveur
  const Status& getStatus() const;

  //! Statistiques du solveur
  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag);

 private:
  bool _solve();
  // A. Anciaux
  bool _solveRs(bool m_resizeable);

  void internalPrintInfo() const;

  void updateParameters();

 private:
  Integer m_max_iteration;
  Real m_stop_criteria_value;
  Integer m_precond_option;
  bool m_precond_pressure;
  bool m_normalisation_pivot;
  bool m_normalize_opt; // pour être homogène avec la configuration de MCGSolver
  Integer m_ilu0_algo;
  bool m_keep_rhs;

  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;
  Status m_status;
  SolverStat m_stat; //<! Statistiques d'exécution du solveur
  SolverStater<IFPInternalLinearSolver> m_stater;
  Integer m_print_info;
  IOptionsIFPLinearSolver* m_options;
};

} // namespace Alien

#endif /* PETSCLINEARSOLVER_H_ */
