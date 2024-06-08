// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalPluginTesterModule.cc                               (C) 2000-2024 */
/*                                                                           */
/* Module de test des plugins externes.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicModule.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/ServiceInfo.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/IExternalPlugin.h"

#include "arcane/tests/ExternalPluginTester_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest::ExternalPluginTester
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module hydrodynamique simplifié.
 *
 * Ce module implémente une hydrodynamique simple tri-dimensionnel,
 * parallèle, avec une pseudo-viscosité aux mailles.
 */
class ExternalPluginTesterModule
: public ArcaneExternalPluginTesterObject
{
 public:

  explicit ExternalPluginTesterModule(const ModuleBuildInfo& cb);
  ~ExternalPluginTesterModule();

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void build() override;
  void init() override;
  void exit() override;
  void computeLoop() override;

 public:

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 1); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExternalPluginTesterModule::
ExternalPluginTesterModule(const ModuleBuildInfo& mb)
: ArcaneExternalPluginTesterObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExternalPluginTesterModule::
staticInitialize(ISubDomain* sd)
{
  // Enregistre la boucle en temps associée à ce module
  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop("ExternalPluginTesterLoop");
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("ExternalPluginTester.EPT_Build"));
    time_loop->setEntryPoints(ITimeLoop::WBuild, clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("ExternalPluginTester.EPT_Init"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("ExternalPluginTester.EPT_ComputeLoop"));
    time_loop->setEntryPoints(String(ITimeLoop::WComputeLoop), clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("ExternalPluginTester.EPT_Exit"));
    time_loop->setEntryPoints(String(ITimeLoop::WExit), clist);
  }
  {
    StringList clist;
    clist.add("ExternalPluginTester");
    time_loop->setRequiredModulesName(clist);
  }
  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExternalPluginTesterModule::
~ExternalPluginTesterModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExternalPluginTesterModule::
build()
{
}

void ExternalPluginTesterModule::
init()
{
  info() << "Begin of init";
  options()->externalPlugin()->loadFile(options()->file());
}

void ExternalPluginTesterModule::
computeLoop()
{
  options()->externalPlugin()->loadFile(options()->file());
  bool is_finished = (m_global_iteration() >= 15);
  if (is_finished)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExternalPluginTesterModule::
exit()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(ExternalPluginTesterModule, ExternalPluginTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest::ExternalPluginTester

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
