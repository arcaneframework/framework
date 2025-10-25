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
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IItemFamily.h"

#include "arcane/tests/MeshReaderUnitTest_axl.h"

#include <random>

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
    c.enableConnectivity(Connectivity::CT_FullConnectivity3D);
  }
  if (!options()->compactMesh()) {
    Properties* p = mesh()->properties();
    p->setBool("compact",false);
    p->setBool("compact-after-allocate",false);
    p->setBool("dump",false);
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
  if (options()->swapNodesUniqueId()) {
    
    // Cela ne fonctionne qu'en séquentiel.
    IItemFamily* node_family = mesh()->nodeFamily();
    Int32 nb_node = node_family->allItems().size();
    info() << "Swapping uniqueIds() of nodes nb_node=" << nb_node;
    NodeInfoListView nodes(node_family);
    std::minstd_rand generator(42);
    std::uniform_int_distribution<int> distribution(0, nb_node-1);
    Int32 nb_swap = nb_node / 2;
    // Génère deux nombres aléatoires et échange les uniqueId
    // des deux noeuds ayant comme localId() ceux qui sont générés
    
    for (Int32 i = 0; i < nb_swap; ++i) {
      Int32 lid0 = distribution(generator);
      Int32 lid1 = distribution(generator);
      Node node0 = nodes[lid0];
      Node node1 = nodes[lid1];
      Int64 uid0 = node0.uniqueId();
      Int64 uid1 = node1.uniqueId();
      info() << "Swap unique ids uid0=" << uid0 << " uid1=" << uid1;
      node0.mutableItemBase().setUniqueId(uid1);
      node1.mutableItemBase().setUniqueId(uid0);
    }
   mesh()->utilities()->recomputeItemsUniqueIdFromNodesUniqueId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
