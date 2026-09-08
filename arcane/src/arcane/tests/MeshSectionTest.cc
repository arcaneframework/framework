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

  void _initVars();

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
  _initVars();
  times.add(m_global_time());

  ServiceBuilder<IMeshSection> spp0(mesh()->handle());
  Ref<IMeshSection> pp0 = spp0.createReference("MeshSection");

  for (auto plane : options()->plane()) {
    pp0->addPlane(plane->p0(), plane->normal());
  }

  VariableCollection vc;
  vc.add(m_on_cells0);
  vc.add(m_on_nodes0);
  vc.add(m_on_faces0);
  vc.add(m_on_cells1);
  vc.add(m_on_nodes1);
  vc.add(m_on_faces1);

  pp0->setVariables(vc);

  pp0->updateSection();
  MeshHandle meshsh = pp0->meshSection();
  IMesh* meshs = meshsh.mesh();


  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(meshsh);
    Ref<IPostProcessorWriter> pp = spp.createReference("Ensight7PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(meshs->nodesCoordinates().variable());
    VariableCollection cloned_var = pp0->variables();
    for (VariableCollection::Enumerator i(cloned_var); ++i;) {
      variables.add(*i);
    }
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(meshs->allNodes());
    groups.add(meshs->allCells());
    post_processor->setGroups(groups);

    IVariableMng* vm = meshs->variableMng();
    vm->writePostProcessing(post_processor);
  }

  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(mesh()->handle());
    Ref<IPostProcessorWriter> pp = spp.createReference("Ensight7PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost2");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(mesh()->nodesCoordinates().variable());
    variables.add(m_on_cells0);
    variables.add(m_on_nodes0);
    variables.add(m_on_faces0);
    variables.add(m_on_cells1);
    variables.add(m_on_nodes1);
    variables.add(m_on_faces1);
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(mesh()->allNodes());
    groups.add(mesh()->allCells());
    post_processor->setGroups(groups);

    IVariableMng* vm = mesh()->variableMng();
    vm->writePostProcessing(post_processor);
  }
}
void MeshSectionTest::
_initVars()
{
  m_on_cells1.resize(3);
  m_on_nodes1.resize(3);
  m_on_faces1.resize(3);

  ENUMERATE_(Cell, icell, mesh()->allCells()){
    m_on_cells0[icell] = icell->uniqueId().asInt32();
    for (Integer i = 0; i < 3; ++i) {
      m_on_cells1[icell][i] = icell->uniqueId().asInt32() * i;
    }
  }

  ENUMERATE_(Node, inode, mesh()->allNodes()){
    m_on_nodes0[inode] = inode->uniqueId().asInt32();
    for (Integer i = 0; i < 3; ++i) {
      m_on_nodes1[inode][i] = inode->uniqueId().asInt32() * i;
    }
  }

  ENUMERATE_(Face, ifaces, mesh()->allFaces()){
    m_on_faces0[ifaces] = ifaces->uniqueId().asInt32();
    for (Integer i = 0; i < 3; ++i) {
      m_on_faces1[ifaces][i] = ifaces->uniqueId().asInt32() * i;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
