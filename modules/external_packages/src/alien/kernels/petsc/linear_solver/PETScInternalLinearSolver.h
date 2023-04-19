#ifndef ALIEN_KERNELS_PETSC_PETSCLINEARSOLVER_H
#define ALIEN_KERNELS_PETSC_PETSCLINEARSOLVER_H

#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/utils/ObjectWithTrace.h>

#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IOptionsPETScLinearSolver;

namespace Alien {
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPETScKSP;
class IPETScPC;
class SolverStat;
class PETScMatrix;
class PETScVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PETScInternalLinearSolver : public IInternalLinearSolver<PETScMatrix, PETScVector>,
                                  public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;

 public:
  struct VerboseTypes
  {
    enum eChoice
    {
      none,
      low,
      high
    };
  };

  struct InitType
  {
    enum eInit
    {
      User,
      Zero,
      Knoll
    };
  };

 public:
  class SolverFactory;
  friend class SolverFactory;

  class PreconditionerFactory;
  friend class PreconditionerFactory;

  class FieldSplitFactory;
  friend class FieldSplitFactory;

 public:
  PETScInternalLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsPETScLinearSolver* options = nullptr);

  virtual ~PETScInternalLinearSolver();

 public:
  //! Initialisation
  void init(int argv, char** argc);
  virtual void init();

  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);

  //! Finalize
  void end();

  /////////////////////////////////////////////////////////////////////////////
  //
  // NEW INTERFACE
  //

  //! return package back end name
  Arccore::String getBackEndName() const { return "petsc"; }

  bool solve(const PETScMatrix& A, const PETScVector& b, PETScVector& x);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  //! Algèbre linéaire compatible
  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Status& getStatus() const;

  //! Statistiques du solveur
  const SolverStat& getSolverStat() const { return m_stat; }

  SolverStat& getSolverStat() { return m_stat; }

  void internalPrintInfo() const;

  bool isParallel()
  {
    if (m_parallel_mng)
      return m_parallel_mng->commSize() > 1;
    else
      return false;
  }

 public:
  void checkError(const Arccore::String& msg, int ierr);

  Arccore::String convergedReasonString(const Arccore::Integer reason) const;

 private:
  bool _isNull(const PETScVector& b);
  bool _solveNullRHS(PETScVector& x);

 public:
  //! Status
  Status m_status;

  //! Indicateur de l'initialisation globale de PETSc
  /*! La vision service de Arcane perd le sens de l'initialisation
   * globale de PETSc.
   * Cette variable globale permet d'y pallier */
  static bool m_global_initialized;

  //! Indicateur du trace-info global pour PETSc
  static bool m_global_want_trace;

  VerboseTypes::eChoice m_verbose;

  //! option to manage null space constant
  bool m_null_space_constant_opt;

  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;

  //! Statistiques d'exécution du solveur
  SolverStat m_stat;

  IOptionsPETScLinearSolver* m_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_PETSC_PETSCLINEARSOLVER_H */
