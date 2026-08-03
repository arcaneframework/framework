// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSectionTest.cc                                          (C) 2000-2026 */
/*                                                                           */
/* MeshSection test service.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ServiceBuilder.h"

#include "arcane/tests/MeshSectionTest_axl.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshSectionTest
: public ArcaneMeshSectionTestObject
{
 public:

  explicit MeshSectionTest(const ServiceBuildInfo& sbi);
  ~MeshSectionTest() override = default;

 public:

  void initializeTest() override;
  void executeTest() override;

private:

  UniqueArray<Real> times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHSECTIONTEST(MeshSectionTest, MeshSectionTest);


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshSectionTest::
MeshSectionTest(const ServiceBuildInfo& sbi)
: ArcaneMeshSectionTestObject(sbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionTest::
executeTest()
{
  times.add(m_global_time());

  ServiceBuilder<IMeshSection> spp0(mesh()->handle());
  Ref<IMeshSection> pp0 = spp0.createReference("MeshSection");

  for (auto plane : options()->plane()) {
    pp0->addPlane(plane->p0(), plane->normal());
  }

  pp0->updateSection();
  MeshHandle meshsh = pp0->meshSection();
  IMesh* meshs = meshsh.mesh();


  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(meshsh);
    Ref<IPostProcessorWriter> pp = spp.createReference("VtkHdfV2PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(meshs->nodesCoordinates().variable());
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(meshs->allNodes());
    post_processor->setGroups(groups);

    IVariableMng* vm = meshs->variableMng();
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
