// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleMaster.h                                              (C) 2000-2025 */
/*                                                                           */
/* Master Module.                                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MODULEMASTER_H
#define ARCANE_CORE_MODULEMASTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"

#include "arcane/core/IModuleMaster.h"
#include "arcane/core/AbstractModule.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Main module.
 *
 * This module is always loaded first so that these entry points encompass
 * all those of other modules.
 * It contains the global case variables, such as the filename or the
 * iteration number.
 */
class ARCANE_CORE_EXPORT ModuleMaster
: public AbstractModule
, public CommonVariables
, public IModuleMaster
{
 public:

  //! Constructor
  explicit ModuleMaster(const ModuleBuildInfo&);

  //! Destructor
  ~ModuleMaster() override;

 public:

  //! Module version
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

  //! Access to module options
  CaseOptionsMain* caseoptions() override { return m_case_options_main; }

  //! Conversion to \a IModule
  IModule* toModule() override { return this; }

  //! Access to 'common' variables shared between all services and modules
  CommonVariables* commonVariables() override { return this; }

  void addTimeLoopService(ITimeLoopService* tls) override;

  //! Dumps standard curves (CPUTime, ElapsedTime, ...)
  void dumpStandardCurves() override;

 public:

  //! Auto-loaded entry point at the beginning of the calculation loop iteration
  /*!
    <ul>
    <li>If the current time is strictly greater than the limit time, requests the termination of the calculation</li>
    <li>Adds the delta calculated in the previous time step to the current time</li>
    </ul>
  */
  void timeLoopBegin();

  //! Auto-loaded entry point at the end of the calculation loop iteration
  /*!
   <ul>
   <li>Increments the iteration counter</li>
   </ul>
  */
  void timeLoopEnd();

  //! Auto-loaded entry point at the beginning of initialization
  void masterInit();

  //! Auto-loaded entry point at the beginning of a new case initialization
  /*! Is not called in case of initialization on a restart */
  void masterStartInit();

  //! Auto-loaded entry point at the beginning of a new case restart
  void masterContinueInit();

 protected:

  //! Overridable time step incrementation
  // note: IFPEN has a notion of event. We can know the next
  // time and time step. If we apply the default incrementation,
  // we get rounding errors...
  virtual void timeIncrementation();

  //! Overridable display of time step information
  // note: IFPEN desires displays configurable per application
  virtual void timeStepInformation();

  void _masterBeginLoop();
  void _masterEndLoop();
  void _masterStartInit();
  void _masterContinueInit();
  void _masterLoopExit();
  void _masterMeshChanged();
  void _masterRestore();

 protected:

  //! Instance of module options
  CaseOptionsMain* m_case_options_main = nullptr;

  //! Number of calculation loops performed
  Integer m_nb_loop = 0;

  //! CPU time value at the last iteration
  Real m_old_cpu_time = 0.0;

  //! Clock time value at the last iteration
  Real m_old_elapsed_time = 0.0;

  //! List of time loop services
  UniqueArray<ITimeLoopService*> m_timeloop_services;

  //! Indicates if we are in the first execution loop
  bool m_is_first_loop = true;

  Real m_thm_mem_used = 0.0;
  Real m_thm_diff_cpu = 0.0;
  Real m_thm_global_cpu_time = 0.0;
  Real m_thm_diff_elapsed = 0.0;
  Real m_thm_global_elapsed_time = 0.0;
  Real m_thm_global_time = 0.0;

  bool m_has_thm_dump_at_iteration = false;

 private:

  void _dumpTimeInfo();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
