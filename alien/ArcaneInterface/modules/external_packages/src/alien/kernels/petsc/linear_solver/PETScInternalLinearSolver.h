// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <vector>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/core/backend/BackEnd.h>
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
class PETScLibrary
: public ObjectWithTrace
{
  public :
  PETScLibrary(std::vector<Arccore::String> const& petsc_options,
               bool use_trace,
               bool exec_on_device,
               bool use_device_momory) ;
  virtual ~PETScLibrary() ;

  BackEnd::Memory::eType getMemoryType() const {
    return m_memory_type ;
  }

  BackEnd::Exec::eSpaceType getExecSpace() const {
    return m_exec_space  ;
  }

  private:
  BackEnd::Memory::eType m_memory_type = BackEnd::Memory::Host ;
  BackEnd::Exec::eSpaceType m_exec_space = BackEnd::Exec::Host ;

  //! Indicateur de l'initialisation globale de PETSc
  /*! La vision service de Arcane perd le sens de l'initialisation
   * globale de PETSc.
   * Cette variable globale permet d'y pallier */
  bool m_global_initialized = false;

  //! Indicateur du trace-info global pour PETSc
  bool m_global_want_trace= false;

} ;

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScInternalLinearSolver
: public IInternalLinearSolver<PETScMatrix, PETScVector>
, public ObjectWithTrace
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


  static bool m_library_plugin_is_initialized ;

  static std::unique_ptr<PETScLibrary> m_library_plugin ;

  static void initializeLibrary(bool exec_on_device=false, bool use_device_momory=false) ;

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


  void setNullSpaceConstantOption(bool flag)
  {
    m_null_space_constant_opt = flag ;
  }

  void setNearNullSpaceOption(bool flag)
  {
    m_nearnull_space_opt = flag ;
  }

  void startNonLinear() final {}

 public:
  void checkError(const Arccore::String& msg, int ierr);

  Arccore::String convergedReasonString(const Arccore::Integer reason) const;

 private:
  bool _isNull(const PETScVector& b);
  bool _solveNullRHS(PETScVector& x);

 public:
  //! Status
  Status m_status;

  VerboseTypes::eChoice m_verbose;

  //! option to manage null space constant
  bool m_null_space_constant_opt = false;
  bool m_nearnull_space_opt = false;

  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng;

  //! Statistiques d'exécution du solveur
  SolverStat m_stat;

  IOptionsPETScLinearSolver* m_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/

