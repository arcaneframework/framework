#ifndef ALIEN_KERNELS_MTL_LINEARSOLVER_MTLINTERNALLINEARSOLVER_H_
#define ALIEN_KERNELS_MTL_LINEARSOLVER_MTLINTERNALLINEARSOLVER_H_

#include <alien/utils/Precomp.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/kernels/mtl/data_structure/MTLInternal.h>
#include <alien/kernels/mtl/linear_solver/MTLOptionTypes.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/data/Space.h>

class IOptionsMTLLinearSolver;

namespace Alien {

class SolverStat;
class MTLMatrix;
class MTLVector;

class MTLInternalLinearSolver : public IInternalLinearSolver<MTLMatrix, MTLVector>,
                                public ObjectWithTrace
{
 private:
  typedef IMatrix MatrixType;
  typedef IVector VectorType;
  typedef IVector Vector;

  typedef SolverStatus Status;

  typedef MTL4Internal::MatrixInternal MatrixInternal;
  typedef MTL4Internal::VectorInternal VectorInternal;

 public:
  // Constructeur de la classe
  MTLInternalLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      IOptionsMTLLinearSolver* options);

  // Destructeur de la classe
  virtual ~MTLInternalLinearSolver();

 public:
  //! Initialisation
  virtual void init();
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);
  void updateParameters();

  //! Finalize
  void end();

  /////////////////////////////////////////////////////////////////////////////
  //
  // NEW INTERFACE
  //
  Arccore::String getBackEndName() const { return "mtl"; }

  //! Résolution du système linéaire
  bool solve(MTLMatrix const& matrix, MTLVector const& rhs, MTLVector& sol);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const;

  //! Algèbre linéaire compatible
  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const;

  //! Statistiques du solveur
  SolverStat& getSolverStat() { return m_stat; }
  const SolverStat& getSolverStat() const { return m_stat; }

 private:
  bool _solve(MatrixInternal::MTLMatrixType const& A,
      VectorInternal::MTLVectorType const& b, VectorInternal::MTLVectorType& x);
  void internalPrintInfo() const;

 private:
  //! Indicateur d'initialisation
  bool m_initialized = false;

  Status m_status;

  MTLOptionTypes::eSolver m_solver_option;
  MTLOptionTypes::ePreconditioner m_preconditioner_option;
  Arccore::Integer m_max_iteration;
  Arccore::Real m_precision;

  int m_output_level = 0;

  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;

  //! Statistiques d'exécution du solveur
  SolverStat m_stat;

  IOptionsMTLLinearSolver* m_options = nullptr;
};

} // namespace Alien

#endif
