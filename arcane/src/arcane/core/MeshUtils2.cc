// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils2.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Various utility functions on the mesh.                                    */
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
      // Adds the entities to the node.
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

    // Set of nodes connected to a node
    // We use a 'std::set' because we want to sort by increasing localId()
    // to always have the same order.
    std::set<Int32> node_set;

   private:

    //! Working array of nodes connected to another node
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

  // To create the connectivity, iterate over all connected cells
  // nodes and then the set of edges of this cell.
  // If one of the two nodes of the edge is my node, add it
  // to the connectivity.
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
    // Adds the entities to the node.
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
  // First, add to boundary_node_map the list of nodes
  // located on the boundary faces.
  ENUMERATE_ (Face, iface, mesh->allFaces()) {
    Face face = *iface;
    Int32 nb_cell = face.nbCell();
    if (nb_cell == 2) {
      Int32 nb_own = 0;
      // If two connected cells, only one of the owners
      // of these two cells is mine (otherwise it is an internal face and thus we
      // do not process it).
      if (face.cell(0).owner() == my_rank)
        ++nb_own;
      if (face.cell(1).owner() == my_rank)
        ++nb_own;
      if (nb_own != 1)
        continue;
    }
    // If only one cell, the face must belong to me.
    if (!face.isOwn())
      continue;
    for (NodeLocalId node_id : face.nodeIds())
      boundary_node_map.add(node_id, true);
  }
  // Now, iterate over the boundary nodes.
  // For each one, iterate over the edges of the cells connected to this node
  // and add the corresponding node if it is also on the boundary
  // (in this case, it is also in boundary_node_map).
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
