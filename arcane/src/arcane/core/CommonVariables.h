// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonVariables.h                                           (C) 2000-2025 */
/*                                                                           */
/* Common variables describing a case.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_COMMONVARIABLES_H
#define ARCANE_CORE_COMMONVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ModuleMaster;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Common variables of a case.
 */
class ARCANE_CORE_EXPORT CommonVariables
{
 public:

  friend class ModuleMaster;

 public:

  //! Constructs the references of the common variables for the module \a c
  CommonVariables(IModule* c);
  //! Constructs the references of the common variables for the manager \a variable_mng
  CommonVariables(IVariableMng* variable_mng);
  // TODO: make deprecated
  //! Constructs the references of the common variables for the subdomain \a sd
  CommonVariables(ISubDomain* sd);
  virtual ~CommonVariables() {} //!< Releases resources.

 public:

  //! Current iteration number
  Int32 globalIteration() const;
  //! Current time
  Real globalTime() const;
  //! Previous current time.
  Real globalOldTime() const;
  //! Final time of the simulation
  Real globalFinalTime() const;
  //! Current Delta T.
  Real globalDeltaT() const;
  //! CPU time used (in seconds)
  Real globalCPUTime() const;
  //! Previous CPU time used (in seconds)
  Real globalOldCPUTime() const;
  //! Clock time (elapsed) used (in seconds)
  Real globalElapsedTime() const;
  //! Previous clock time (elapsed) used (in seconds)
  Real globalOldElapsedTime() const;

 private:
 public:

  VariableScalarInt32 m_global_iteration; //!< Current iteration
  VariableScalarReal m_global_time; //!< Current time
  VariableScalarReal m_global_deltat; //!< Global Delta T
  VariableScalarReal m_global_old_time; //!< Time previous to the current time
  VariableScalarReal m_global_old_deltat; //!< Delta T at the time previous to the global time
  VariableScalarReal m_global_final_time; //!< Final time of the case
  VariableScalarReal m_global_old_cpu_time; //!< Previous CPU time used (in seconds)
  VariableScalarReal m_global_cpu_time; //!< CPU time used (in seconds)
  VariableScalarReal m_global_old_elapsed_time; //!< Previous clock time used (in seconds)
  VariableScalarReal m_global_elapsed_time; //!< Clock time used (in seconds)
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
