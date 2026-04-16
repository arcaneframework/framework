// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HydroAdditionalTestModule.cc                                 C) 2000-2026 */
/*                                                                           */
/* Tests Additionnels couplés au module Hydro.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/tests/HydroAdditionalTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HydroAdditionalTestModule
: public ArcaneHydroAdditionalTestObject
{
 public:

  explicit HydroAdditionalTestModule(const ModuleBuildInfo& sbi)
  : ArcaneHydroAdditionalTestObject(sbi)
  {}

 public:

  void init() override;
  void doIterationEnd() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HydroAdditionalTestModule::
init()
{
  // Créé une connectivité noeud->noeud contenant pour chaque noeuds les
  // noeuds des mailles connectés à ce noeud ainsi que lui-même.
  IItemFamily* node_family = mesh()->nodeFamily();
  NodeGroup nodes = node_family->allItems();
  // NOTE: l'objet est automatiquement détruit par le maillage
  auto* idxm = mesh()->indexedConnectivityMng();
  auto idx_cn = idxm->findOrCreateConnectivity(node_family, node_family, "NodeCellNode");
  auto* cn = idx_cn->connectivity();
  std::set<ItemLocalId> done_set;
  ENUMERATE_ (Node, inode, nodes) {
    done_set.clear();
    Node node = *inode;
    done_set.insert(node);
    for (Cell cell : node.cells()) {
      for (NodeLocalId sub_node : cell.nodes()) {
        done_set.insert(sub_node);
      }
    }
    for (auto x : done_set)
      cn->addConnectedItem(node, x);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HydroAdditionalTestModule::
doIterationEnd()
{
  auto idx_cn = mesh()->indexedConnectivityMng()->findConnectivity("NodeCellNode");
  IItemFamily* node_family = mesh()->nodeFamily();
  NodeGroup nodes = node_family->allItems();
  IndexedNodeNodeConnectivityView cn_view = idx_cn->view();
  NodeInfoListView nodes_view(node_family);
  ENUMERATE_ (Node, inode, nodes) {
    if (inode.itemLocalId() > 5)
      break;
    Node node = *inode;
    info() << " Node=" << ItemPrinter(node) << " n=" << cn_view.nbNode(node);
    for (NodeLocalId sub_node : cn_view.nodes(node)) {
      info() << " I=" << ItemPrinter(nodes_view[sub_node]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_HYDROADDITIONALTEST(HydroAdditionalTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
