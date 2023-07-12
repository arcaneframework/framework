/*
 * HypreInternalLinearSolver.h
 *
 *  Created on: 30 avr. 2015
 *      Author: chevalic
 */

#ifndef ALIEN_KERNELS_HYPRE_LINEARSOLVER_HYPREINTERNALLINEARSOLVER_H
#define ALIEN_KERNELS_HYPRE_LINEARSOLVER_HYPREINTERNALLINEARSOLVER_H

#include <memory>

#include <alien/utils/Precomp.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IOptionsHypreSolver;

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SolverStat;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HypreMatrix;
class HypreVector;

class HypreLibrary
{
  public :
  HypreLibrary() ;
  virtual ~HypreLibrary() ;
} ;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HypreInternalLinearSolver : public IInternalLinearSolver<HypreMatrix, HypreVector>,
                                  public ObjectWithTrace
{
 public:
  typedef SolverStatus Status;

  HypreInternalLinearSolver(Arccore::MessagePassing::IMessagePassingMng* pm = nullptr,
      IOptionsHypreSolver* options = nullptr);

  virtual ~HypreInternalLinearSolver();

 public:

  static bool m_library_plugin_is_initialized ;

  static std::unique_ptr<HypreLibrary> m_library_plugin ;

  virtual void init();

  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);

  void end();

  Arccore::String getBackEndName() const { return "hypre"; }

  //! Résolution du système linéaire
  bool solve(const HypreMatrix& A, const HypreVector& b, HypreVector& x);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Status& getStatus() const;
  Status& getStatusRef() { return m_status; }

  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

 private:
  Status m_status;

  Integer m_gpu_device_id = 0 ;

  SolverStat m_stat;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;
  IOptionsHypreSolver* m_options;

 private:
  void checkError(const Arccore::String& msg, int ierr, int skipError = 0) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_HYPRE_LINEARSOLVER_HYPREINTERNALLINEARSOLVER_H_ */
