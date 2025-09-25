// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshReaderUnitTest.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Service de test de lecture du maillage.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Properties.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/IMeshUniqueIdMng.h"

#include "arcane/tests/MeshReaderUnitTest_axl.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage
 */
class MeshReaderUnitTest
: public ArcaneMeshReaderUnitTestObject
{
public:

  explicit MeshReaderUnitTest(const ServiceBuildInfo& sbi);

public:

  void buildInitializeTest() override;
  void executeTest() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHREADERUNITTEST(MeshReaderUnitTest,MeshReaderUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshReaderUnitTest::
MeshReaderUnitTest(const ServiceBuildInfo& sbi)
: ArcaneMeshReaderUnitTestObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshReaderUnitTest::
buildInitializeTest()
{
  if (options()->createEdges()) {
    Connectivity c(mesh()->connectivity());
    c.enableConnectivity(Connectivity::CT_HasEdge);
  }
  if (!options()->compactMesh()) {
    Properties* p = mesh()->properties();
    p->setBool("compact",false);
    p->setBool("compact-after-allocate",false);
  }
  if (options()->generateUidFromNodesUid()) {
    mesh()->meshUniqueIdMng()->setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshReaderUnitTest::
executeTest()
{
  info() << "Execute Test";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
