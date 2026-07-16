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

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/ServiceBuilder.h"

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

  explicit MeshCutTest(const ServiceBuildInfo& sbi);
  ~MeshCutTest() override = default;

 public:

  void initializeTest() override;
  void executeTest() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHCUTTEST(MeshCutTest, MeshCutTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCutTest::
MeshCutTest(const ServiceBuildInfo& sbi)
: ArcaneMeshCutTestObject(sbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
executeTest()
{
  ServiceBuilder<IMeshSection> spp0(mesh()->handle());
  Ref<IMeshSection> pp0 = spp0.createReference("MeshCutService");

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
