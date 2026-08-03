// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCutTest.cc                                              (C) 2000-2026 */
/*                                                                           */
/* MeshCut test service.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/Directory.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"

#include "arcane/tests/MeshCutTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshCutTest
: public ArcaneMeshCutTestObject
{
 public:

  explicit MeshCutTest(const ModuleBuildInfo& mbi);
  ~MeshCutTest() override = default;

public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void compute() override;

 private:

  UniqueArray<Real> times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MESHCUTTEST(MeshCutTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCutTest::
MeshCutTest(const ModuleBuildInfo& mbi)
: ArcaneMeshCutTestObject(mbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("MeshCutTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshCutTest.Compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("MeshCutTest");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
compute()
{
  {
    m_global_time = globalIteration();
    times.add(m_global_time());
  }

  //
  // MeshHandle meshhsection;
  // {
  //   ServiceBuilder<IMeshSection> spp0(mesh()->handle());
  //   Ref<IMeshSection> pp0 = spp0.createReference("MeshSection");
  //
  //   pp0->addPlane({0.95, 0, 0}, {1, 0, 0});
  //   pp0->addPlane({0.97, 0, 0}, {-1, 0, 0});
  //
  //   pp0->updateSection();
  //   meshhsection = pp0->meshSection();
  // }
  // IMesh* meshsection = meshhsection.mesh();
  //
  // if (options()->enablePostProcessing())
  // {
  //   ServiceBuilder<IPostProcessorWriter> spp(meshhsection);
  //   Ref<IPostProcessorWriter> pp = spp.createReference("VtkHdfV2PostProcessor");
  //   Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
  //   pp->setBaseDirectoryName(output_directory.path());
  //   IPostProcessorWriter* post_processor = pp.get();
  //   post_processor->setTimes(times);
  //
  //   VariableList variables;
  //   variables.add(meshsection->nodesCoordinates().variable());
  //   post_processor->setVariables(variables);
  //
  //   ItemGroupList groups;
  //   groups.add(meshsection->allNodes());
  //   post_processor->setGroups(groups);
  //
  //   IVariableMng* vm = meshsection->variableMng();
  //   vm->writePostProcessing(post_processor);
  // }


  MeshHandle meshhcut;
  {
    ServiceBuilder<IMeshSection> spp0(mesh()->handle());
    // ServiceBuilder<IMeshSection> spp0(meshhsection);
    Ref<IMeshSection> pp0 = spp0.createReference("MeshCut");

    for (auto plane : options()->plane()) {
      pp0->addPlane(plane->p0() + (plane->p0Velocity() * globalIteration()), plane->normal());
    }

    pp0->updateSection();
    meshhcut = pp0->meshSection();
  }
  IMesh* meshcut = meshhcut.mesh();


  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(meshhcut);
    Ref<IPostProcessorWriter> pp = spp.createReference("VtkHdfV2PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(meshcut->nodesCoordinates().variable());
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(meshcut->allNodes());
    post_processor->setGroups(groups);

    IVariableMng* vm = meshcut->variableMng();
    vm->writePostProcessing(post_processor);
  }

  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(mesh()->handle());
    Ref<IPostProcessorWriter> pp = spp.createReference("VtkHdfV2PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(mesh()->nodesCoordinates().variable());
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(mesh()->allNodes());
    post_processor->setGroups(groups);

    IVariableMng* vm = mesh()->variableMng();
    vm->writePostProcessing(post_processor);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
