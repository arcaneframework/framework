// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils2.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires diverses sur le maillage.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshUtils.h"

#include "arcane/utils/SmallArray.h"
#include "arcane/utils/HashTableMap2.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IParallelMng.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  class NodeNodeConnectivityHelper
  {
   public:

    NodeNodeConnectivityHelper(IMesh* mesh, const String& connectivity_name)
    : node_family(mesh->nodeFamily())
    {
      auto connectivity_mng = mesh->indexedConnectivityMng();
      connectivity_ref = connectivity_mng->findOrCreateConnectivity(node_family, node_family, connectivity_name);
      item_connectivity = connectivity_ref->connectivity();
    }

   public:

    void addCurrentNodeSetToNode(NodeLocalId node_lid)
    {
      // Ajoute les entités au nœud.
      {
        Int32 nb_node = CheckedConvert::toInt32(node_set.size());
        if (nb_node == 0)
          return;
        connected_items_ids.resize(nb_node);
        Int32 index = 0;
        for (auto x : node_set) {
          connected_items_ids[index] = x;
          ++index;
        }
        item_connectivity->setConnectedItems(node_lid, connected_items_ids);
      }
      node_set.clear();
    }

   public:

    IItemFamily* node_family = nullptr;
    Ref<IIndexedIncrementalItemConnectivity> connectivity_ref;

    // Ensemble des nœuds connectés à un nœud
    // On utilise un 'std::set' car on veut trier par localId() croissant
    // pour avoir toujours le même ordre.
    std::set<Int32> node_set;

   private:

    //! Tableau de travail des noeuds connectés à un autre noeud
    SmallArray<Int32> connected_items_ids;

    IIncrementalItemConnectivity* item_connectivity = nullptr;
  };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIndexedIncrementalItemConnectivity> MeshUtils::
computeNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name)
{
  ARCANE_CHECK_POINTER(mesh);
  NodeNodeConnectivityHelper helper(mesh, connectivity_name);

  // Pour créer la connectivité, parcours l'ensemble des mailles connectées
  // nœud et ensuite l'ensemble des arêtes de cette maille.
  // Si un des deux nœuds de l'arête est mon nœud, l'ajoute
  // à la connectivité.
  ENUMERATE_ (Node, inode, helper.node_family->allItems()) {
    Node node = *inode;
    NodeLocalId node_lid(node.localId());
    for (Cell cell : node.cells()) {
      const ItemTypeInfo* t = cell.typeInfo();
      for (Int32 i = 0, n = t->nbLocalEdge(); i < n; ++i) {
        ItemTypeInfo::LocalEdge e = t->localEdge(i);
        NodeLocalId node0_lid = cell.nodeId(e.beginNode());
        NodeLocalId node1_lid = cell.nodeId(e.endNode());
        if (node0_lid == node_lid)
          helper.node_set.insert(node1_lid);
        if (node1_lid == node_lid)
          helper.node_set.insert(node0_lid);
      }
    }
    // Ajoute les entités au nœud.
    helper.addCurrentNodeSetToNode(node_lid);
  }

  return helper.connectivity_ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIndexedIncrementalItemConnectivity> MeshUtils::
computeBoundaryNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name)
{
  ARCANE_CHECK_POINTER(mesh);
  ARCANE_CHECK_POINTER(mesh);
  NodeNodeConnectivityHelper helper(mesh, connectivity_name);

  Arcane::impl::HashTableMap2<Int32, bool> boundary_node_map;
  Int32 my_rank = mesh->parallelMng()->commRank();
  // D'abord, ajoute dans \a boundary_node_map la liste des noeuds
  // situés sur les faces au de bord.
  ENUMERATE_ (Face, iface, mesh->allFaces()) {
    Face face = *iface;
    Int32 nb_cell = face.nbCell();
    if (nb_cell == 2) {
      Int32 nb_own = 0;
      // Si deux mailles connectées, il faut que l'un seulement des propriétaires
      // de ces deux mailles soit le mien (sinon c'est une face interne et donc on
      // ne la traite pas).
      if (face.cell(0).owner() == my_rank)
        ++nb_own;
      if (face.cell(1).owner() == my_rank)
        ++nb_own;
      if (nb_own != 1)
        continue;
    }
    // Si une seule maille, il faut que la face m'appartienne.
    if (!face.isOwn())
      continue;
    for (NodeLocalId node_id : face.nodeIds())
      boundary_node_map.add(node_id, true);
  }
  // Maintenant, parcours les noeuds du bord.
  // Pour chacun parcours les arêtes des mailles connectées à ce nœud
  // et ajoute le nœud correspondant s'il est aussi sur la frontière
  // (dans ce cas, il est aussi dans boundary_node_map).
  NodeLocalIdToNodeConverter nodes(helper.node_family);
  for (auto [node_lid, _] : boundary_node_map) {
    Node node = nodes[node_lid];
    for (Cell cell : node.cells()) {
      const ItemTypeInfo* t = cell.typeInfo();
      for (Int32 i = 0, n = t->nbLocalEdge(); i < n; ++i) {
        ItemTypeInfo::LocalEdge e = t->localEdge(i);
        NodeLocalId node0_lid = cell.nodeId(e.beginNode());
        NodeLocalId node1_lid = cell.nodeId(e.endNode());
        Int32 other_node_lid = NULL_ITEM_LOCAL_ID;
        if (node0_lid == node_lid)
          other_node_lid = node1_lid;
        if (node1_lid == node_lid)
          other_node_lid = node0_lid;
        if (other_node_lid != NULL_ITEM_LOCAL_ID && boundary_node_map.contains(other_node_lid))
          helper.node_set.insert(other_node_lid);
      }
    }
    helper.addCurrentNodeSetToNode(NodeLocalId(node_lid));
  }
  return helper.connectivity_ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
