// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils2.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires diverses sur le maillage.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshUtils.h"

#include "arcane/utils/SmallArray.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/core/IIncrementalItemConnectivity.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIndexedIncrementalItemConnectivity> MeshUtils::
createNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name)
{
  IItemFamily* node_family = mesh->nodeFamily();
  auto connectivity_mng = mesh->indexedConnectivityMng();
  auto connectivity_ref = connectivity_mng->findOrCreateConnectivity(node_family, node_family, connectivity_name);
  IIncrementalItemConnectivity* cx = connectivity_ref->connectivity();

  // Ensemble des nœuds connectés à un nœud
  // On utilise un 'std::set' car on veut trier par localId() croissant
  // pour avoir toujours le même ordre.
  std::set<Int32> node_set;

  // Pour créer la connectivité, parcours l'ensemble des mailles connectées
  // nœud et ensuite l'ensemble des arêtes de cette maille.
  // Si un des deux nœuds de l'arête est mon nœud, l'ajoute
  // à la connectivité.
  SmallArray<Int32> connected_items_ids;
  ENUMERATE_ (Node, inode, node_family->allItems()) {
    Node node = *inode;
    NodeLocalId node_lid(node.localId());
    node_set.clear();
    for (Cell cell : node.cells()) {
      const ItemTypeInfo* t = cell.typeInfo();
      for (Int32 i = 0, n = t->nbLocalEdge(); i < n; ++i) {
        ItemTypeInfo::LocalEdge e = t->localEdge(i);
        NodeLocalId node0_lid = cell.nodeId(e.beginNode());
        NodeLocalId node1_lid = cell.nodeId(e.endNode());
        if (node0_lid == node_lid)
          node_set.insert(node1_lid);
        if (node1_lid == node_lid)
          node_set.insert(node0_lid);
      }
    }
    // Ajoute les entités au noeud.
    {
      connected_items_ids.resize(node_set.size());
      Int32 index = 0;
      for (auto x : node_set) {
        connected_items_ids[index] = x;
        ++index;
      }
      cx->setConnectedItems(node_lid, connected_items_ids);
    }
  }

  return connectivity_ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
