// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include <alien/core/backend/KernelSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
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
  HypreLibrary(bool exec_on_device, bool use_device_momory, int device_id=0) ;
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
  int m_device_id = 0 ;
} ;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT HypreInternalLinearSolver
: public IInternalLinearSolver<HypreMatrix, HypreVector>
, public KernelSolverT<BackEnd::tag::hypre>
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

  static void initializeLibrary(bool exec_on_device=false, bool use_device_momory=false, int device_id=0) ;

  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);

  Arccore::String getBackEndName() const { return "hypre"; }

  void init();

  void init(HypreMatrix const& A) ;

  void start() {

  }

  void end();

  //! Résolution du système linéaire
  bool solve(const HypreMatrix& A, const HypreVector& b, HypreVector& x);

  bool solve(const HypreVector& b, HypreVector& x);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Status& getStatus() const;
  Status& getStatusRef() { return m_status; }

  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

 private:
  struct Impl ;

  std::unique_ptr<Impl> m_impl ;

  Status m_status;

  Integer m_gpu_device_id = 0 ;

  SolverStat m_stat;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;
  IOptionsHypreSolver* m_options;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_HYPRE_LINEARSOLVER_HYPREINTERNALLINEARSOLVER_H_ */
