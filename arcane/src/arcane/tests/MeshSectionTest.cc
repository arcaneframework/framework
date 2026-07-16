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

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IMeshSection.h"

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
  ServiceBuilder<IMeshSection> spp0(mesh()->handle());
  Ref<IMeshSection> pp0 = spp0.createReference("MeshSectionService");

  for (auto plane : options()->plane()) {
    pp0->addPlane(plane->p0(), plane->normal());
  }

  pp0->updateSection();
  MeshHandle meshsh = pp0->meshSection();
  IMesh* meshs = meshsh.mesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
