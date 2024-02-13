// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

#include "TypesSimpleHydro.h"
#include "SimpleHydro_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleHydroModuleBase::getDeltatInit() { return m_options->deltatInit(); }
TypesSimpleHydro::eViscosity SimpleHydroModuleBase::getViscosity(){ return m_options->viscosity(); }
Real SimpleHydroModuleBase::getViscosityLinearCoef(){ return m_options->viscosityLinearCoef(); }
Real SimpleHydroModuleBase::getViscosityQuadraticCoef(){ return m_options->viscosityQuadraticCoef(); }
ConstArrayView<IBoundaryCondition*> SimpleHydroModuleBase::getBoundaryConditions(){ return m_options->getBoundaryCondition(); }
Real SimpleHydroModuleBase::getCfl(){ return m_options->cfl(); }
Real SimpleHydroModuleBase::getVariationSup(){ return m_options->variationSup(); }
Real SimpleHydroModuleBase::getVariationInf(){ return m_options->variationInf(); }
Real SimpleHydroModuleBase::getDensityGlobalRatio(){ return m_options->densityGlobalRatio(); }
Real SimpleHydroModuleBase::getDeltatMax(){ return m_options->deltatMax(); }
Real SimpleHydroModuleBase::getDeltatMin(){ return m_options->deltatMin(); }
Real SimpleHydroModuleBase::getFinalTime(){ return m_options->finalTime(); }
Integer SimpleHydroModuleBase::getBackwardIteration(){ return m_options->backwardIteration(); }
bool SimpleHydroModuleBase::isCheckNumericalResult() { return m_options->checkNumericalResult(); }

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

  void hydroBuild();

  void hydroStartInit() { m_service->hydroStartInit(); }
  void hydroInit() { m_service->hydroInit(); }
  void hydroContinueInit() {}
  void hydroExit();

  void computeForces() { m_service->computeForces(); }
  void computePressureForce(){}
  void computePseudoViscosity(){}
  void computeVelocity() { m_service->computeVelocity(); }
  void computeViscosityWork() { m_service->computeViscosityWork(); }
  void applyBoundaryCondition() { m_service->applyBoundaryCondition(); }
  void moveNodes(){ m_service->moveNodes(); }
  void computeGeometricValues() { m_service->computeGeometricValues(); }
  void updateDensity() { m_service->updateDensity(); }
  void applyEquationOfState() { m_service->applyEquationOfState(); }
  void computeDeltaT(){ m_service->computeDeltaT(); }
  void doOneIteration();

 private:

  Ref<ISimpleHydroService> m_service;
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
  String service_name = options()->genericServiceName();
  if (service_name.empty())
    ARCANE_FATAL("Null or empty <generic-service-name> option");
  info() << "Creating hydro service name=" << service_name;
  m_service = sb.createReference(service_name);
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
  ISimpleHydroService* s = m_service.get();
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
