// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MiniWeatherModule.cc                                        (C) 2000-2021 */
/*                                                                           */
/* Module pour la miniapplication MiniWeather.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/core/BasicModule.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/ServiceInfo.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/tests/MiniWeatherTypes.h"
#include "arcane/tests/MiniWeather_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest::MiniWeather
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
class MiniWeatherModule
: public ArcaneMiniWeatherObject
{
 public:

  explicit MiniWeatherModule(const ModuleBuildInfo& cb);
  ~MiniWeatherModule();

 public:
  
  static void staticInitialize(ISubDomain* sd);

 public:

  void build() override;
  void init() override;
  void exit() override;
  void computeLoop() override;

 public:
	
  VersionInfo versionInfo() const override { return VersionInfo(1,0,1); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MiniWeatherModule::
MiniWeatherModule(const ModuleBuildInfo& mb)
: ArcaneMiniWeatherObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherModule::
staticInitialize(ISubDomain* sd)
{
  // Enregistre la boucle en temps associée à ce module
  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop("MiniWeatherLoop");
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MiniWeather.MW_Build"));
    time_loop->setEntryPoints(ITimeLoop::WBuild,clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MiniWeather.MW_Init"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MiniWeather.MW_ComputeLoop"));
    time_loop->setEntryPoints(String(ITimeLoop::WComputeLoop),clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MiniWeather.MW_Exit"));
    time_loop->setEntryPoints(String(ITimeLoop::WExit),clist);
  }
  {
    StringList clist;
    clist.add("MiniWeather");
    time_loop->setRequiredModulesName(clist);
  }
  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MiniWeatherModule::
~MiniWeatherModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherModule::
build()
{
}

void MiniWeatherModule::
init()
{
  info() << "Begin of init";
  eMemoryRessource memory = eMemoryRessource::UnifiedMemory;
  if (options()->useDeviceMemory()){
    info() << "Using device memory";
    memory = eMemoryRessource::Device;
  }
  info() << "MemoryRessource: " << memory;

  options()->implementation()->init(acceleratorMng(),options()->nbCellX(),
                                    options()->nbCellZ(),options()->finalTime(),memory,options()->useLeftLayout());
}

void MiniWeatherModule::
computeLoop()
{
  bool is_finished = options()->implementation()->loop();
  if (is_finished)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherModule::
exit()
{
  constexpr int NB_VAR = 4;
  UniqueArray<Real> reduced_values(NB_VAR,0.0);
  options()->implementation()->exit(reduced_values);

  double ref_v[NB_VAR] =
  {
    26.6243096397231,
    2631.23267576729,
    -259.490171322721,
    7897.73654775889
  };
  for ( int ll = 0; ll < NB_VAR; ll++)
    info() << "SUM var" << ll << " sum_v=" << reduced_values[ll];

  Integer nb_x = options()->nbCellX();
  Integer nb_y = options()->nbCellZ();
  Real final_time = options()->finalTime();

  // Compare avec la référence (uniquement valide pour x=400,z=200,final_time=2.0)
  if (nb_x==400 && nb_y==200 && final_time==2.0){
    info() << "Compare values with reference";
    for (int ll = 0; ll < NB_VAR; ll++){
      double rv = ref_v[ll];
      double sv = reduced_values[ll];
      Real diff = math::abs((rv-sv)/rv);
      info() << "var=" << ll << " SUM=" << sv << " diff=" << diff;
      // Il faut utiliser un epsilon assez élevé car les différences peuvent
      // être importantes entre X64, ARM ou les GPU.
      if (!math::isNearlyEqualWithEpsilon(sv,rv,1e-12)){
        ARCANE_FATAL("Bad value ref={0} v={1} var={2} diff={3}",rv, sv, ll, diff);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(MiniWeatherModule,MiniWeather);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest::MiniWeather

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
