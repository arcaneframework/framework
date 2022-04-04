﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleSimpleHydroGeneric.cc                                 (C) 2000-2020 */
/*                                                                           */
/* Module Hydrodynamique simple délégant l'implémentation.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/BasicModule.h"
#include "arcane/ITimeLoop.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/EntryPoint.h"
#include "arcane/MathUtils.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/VariableTypes.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IParallelMng.h"
#include "arcane/ModuleFactory.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/ItemPrinter.h"
#include "arcane/Concurrency.h"
#include "arcane/BasicService.h"
#include "arcane/ServiceBuildInfo.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/FactoryService.h"
#include "arcane/ITimeStats.h"

#include "arcane/IMainFactory.h"

#include "arcane/tests/TypesSimpleHydro.h"
#include "arcane/tests/SimpleHydro_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module hydrodynamique simplifié générique
 * utilisant un service pour l'implémentation.
 */
class ModuleSimpleHydroGeneric
: public ArcaneSimpleHydroObject
{
 public:

  //! Constructeur
  explicit ModuleSimpleHydroGeneric(const ModuleBuildInfo& mb);

 public:
  
  VersionInfo versionInfo() const override { return VersionInfo(1,0,1); }

 public:

  void hydroBuild() override;

  void hydroStartInit() override { m_service->hydroStartInit(); }
  void hydroInit() override { m_service->hydroInit(); }
  void hydroContinueInit()  override{}
  void hydroExit() override;

  void computeForces() override { m_service->computeForces(); }
  void computePseudoViscosity() override{}
  void computeVelocity() override { m_service->computeVelocity(); }
  void computeViscosityWork() override { m_service->computeViscosityWork(); }
  void applyBoundaryCondition() override { m_service->applyBoundaryCondition(); }
  void moveNodes() override{ m_service->moveNodes(); }
  void computeGeometricValues() override { m_service->computeGeometricValues(); }
  void updateDensity() override { m_service->updateDensity(); }
  void applyEquationOfState() override { m_service->applyEquationOfState(); }
  void computeDeltaT() override{ m_service->computeDeltaT(); }
  void doOneIteration() override;

 private:

  ISimpleHydroService* m_service = nullptr;
  ITimeStats* m_time_stats;
  Timer m_elapsed_timer;

 private:

  void _doCall(const char* func_name,std::function<void()> func);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleSimpleHydroGeneric::
ModuleSimpleHydroGeneric(const ModuleBuildInfo& mb)
: ArcaneSimpleHydroObject(mb)
, m_time_stats(mb.subDomain()->timeStats())
, m_elapsed_timer(mb.subDomain(),"HydroGeneric",Timer::TimerReal)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroGeneric::
hydroBuild()
{
  _setHydroOptions(options());
  ServiceBuilder<ISimpleHydroService> sb(subDomain());
  ISimpleHydroService* service = options()->genericService();
  if (!service)
    ARCANE_FATAL("Null or empty <generic-service> option");
  m_service = service;
  m_service->setModule(this);
  m_service->hydroBuild();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroGeneric::
hydroExit()
{
  m_service->hydroExit();
  m_time_stats->dumpCurrentStats("SH_DoOneIteration");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroGeneric::
_doCall(const char* func_name,std::function<void()> func)
{
  {
    Timer::Sentry ts_elapsed(&m_elapsed_timer);
    Timer::Action ts_action1(m_time_stats,func_name);
    func();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define DO_CALL(func_name) \
  _doCall(#func_name, [=]{ s-> func_name ();})

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroGeneric::
doOneIteration()
{
  ISimpleHydroService* s = m_service;
  DO_CALL(computeForces);
  DO_CALL(computeVelocity);
  DO_CALL(computeViscosityWork);
  DO_CALL(applyBoundaryCondition);
  DO_CALL(moveNodes);
  DO_CALL(computeGeometricValues);
  DO_CALL(updateDensity);
  DO_CALL(applyEquationOfState);
  DO_CALL(computeDeltaT);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(ModuleSimpleHydroGeneric,SimpleHydroGeneric);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace SimpleHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
