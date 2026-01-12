// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIEN_KERNELS_HYPRE_LINEARSOLVER_HYPREINTERNALLINEARSOLVER_H
#define ALIEN_KERNELS_HYPRE_LINEARSOLVER_HYPREINTERNALLINEARSOLVER_H

#include <memory>

#include <alien/utils/Precomp.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/core/backend/BackEnd.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/AlienExternalPackagesPrecomp.h>

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
  HypreLibrary(bool exec_on_device, bool use_device_momory) ;
  virtual ~HypreLibrary() ;

  BackEnd::Memory::eType getMemoryType() const {
    return m_memory_type ;
  }

  BackEnd::Exec::eSpaceType getExecSpace() const {
    return m_exec_space  ;
  }

  private:
  BackEnd::Memory::eType m_memory_type = BackEnd::Memory::Host ;
  BackEnd::Exec::eSpaceType m_exec_space = BackEnd::Exec::Host ;
} ;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT HypreInternalLinearSolver 
: public IInternalLinearSolver<HypreMatrix, HypreVector>
, public ObjectWithTrace
{
 public:
  typedef SolverStatus Status;

  HypreInternalLinearSolver(Arccore::MessagePassing::IMessagePassingMng* pm = nullptr,
      IOptionsHypreSolver* options = nullptr);

  virtual ~HypreInternalLinearSolver();

 public:

  static bool m_library_plugin_is_initialized ;

  static std::unique_ptr<HypreLibrary> m_library_plugin ;

  static void initializeLibrary(bool exec_on_device=false, bool use_device_momory=false) ;

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

  void startNonLinear() final {}

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
