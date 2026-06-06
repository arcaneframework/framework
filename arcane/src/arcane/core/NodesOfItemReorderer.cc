// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodesOfItemReorderer.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Utility class for reordering the nodes of an entity.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/NodesOfItemReorderer.h"

#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/ItemTypeId.h"
#include "arcane/core/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the order 1 version
// Only implemented for edges
bool NodesOfItemReorderer::
_reorderOrder3(ConstArrayView<Int64> nodes_uid,
               ArrayView<Int64> sorted_nodes_uid,
               [[maybe_unused]] bool has_center_node)
{
  // \a true if faces need to be reoriented so that their orientation
  // is independent of the initial mesh partitioning.

  Int32 nb_node = nodes_uid.size();

  // Only handles the case of order 3 edges, which therefore have 4 nodes
  if (nb_node != 4)
    ARCANE_THROW(NotImplementedException, "Node reordering for 2D type of order 3 or more");

  if (nodes_uid[0] < nodes_uid[1]) {
    // Nothing to do
    sorted_nodes_uid[0] = nodes_uid[0];
    sorted_nodes_uid[1] = nodes_uid[1];
    sorted_nodes_uid[2] = nodes_uid[2];
    sorted_nodes_uid[3] = nodes_uid[3];
    return false;
  }
  sorted_nodes_uid[0] = nodes_uid[1];
  sorted_nodes_uid[1] = nodes_uid[0];
  sorted_nodes_uid[2] = nodes_uid[3];
  sorted_nodes_uid[3] = nodes_uid[2];
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the order 1 version
bool NodesOfItemReorderer::
_reorderOrder2(ConstArrayView<Int64> nodes_uid,
               ArrayView<Int64> sorted_nodes_uid, bool has_center_node)
{
  // \a true if faces need to be reoriented so that their orientation
  // is independent of the initial mesh partitioning.
  bool need_swap_orientation = false;
  Int32 min_node_index = 0;

  Int32 nb_node = nodes_uid.size();

  // Directly handles the case of order 2 edges
  if (nb_node == 3) {
    if (nodes_uid[0] < nodes_uid[1]) {
      sorted_nodes_uid[0] = nodes_uid[0];
      sorted_nodes_uid[1] = nodes_uid[1];
      sorted_nodes_uid[2] = nodes_uid[2];
      return false;
    }
    sorted_nodes_uid[0] = nodes_uid[1];
    sorted_nodes_uid[1] = nodes_uid[0];
    sorted_nodes_uid[2] = nodes_uid[2];
    return true;
  }
  // If there is a center node, it is the last node in the list
  // and it should not be sorted
  // NOTE: In this case, the number of nodes in the entity is odd.
  if (has_center_node)
    sorted_nodes_uid[nb_node - 1] = nodes_uid[nb_node - 1];

  // For order 2, if we have N nodes, we only need to test the first N/2 nodes
  // TODO: use type information.
  nb_node = nb_node / 2;

  // The following algorithm orients the faces by taking into account only
  // the order of the numbering of these nodes. If this order is
  // preserved during partitioning, then the orientation of the faces
  // will also be preserved.

  // The algorithm is as follows:
  // - Finds node n with the smallest index.
  // - Finds n-1 and n+1, the indices of its 2 neighboring nodes.
  // - If (n+1) is less than (n-1), the orientation n is not modified.
  // - If (n+1) is greater than (n-1), the orientation is inverted.

  // Finds the node with the smallest index

  Int64 min_node = INT64_MAX;
  for (Integer k = 0; k < nb_node; ++k) {
    Int64 id = nodes_uid[k];
    if (id < min_node) {
      min_node = id;
      min_node_index = k;
    }
  }
  Int64 next_node = nodes_uid[(min_node_index + 1) % nb_node];
  Int64 prev_node = nodes_uid[(min_node_index + (nb_node - 1)) % nb_node];
  Integer incr = 0;
  Integer incr2 = 0;
  // Tests the case where the previous or next nodes
  // are the same as the node with the smallest uniqueId().
  // (in this case, the entity is semi-degenerate)
  {
    if (next_node == min_node) {
      next_node = nodes_uid[(min_node_index + (nb_node + 2)) % nb_node];
      incr = 1;
    }
    if (prev_node == min_node) {
      prev_node = nodes_uid[(min_node_index + (nb_node - 2)) % nb_node];
      incr2 = nb_node - 1;
    }
  }
  if (next_node > prev_node)
    need_swap_orientation = true;
  if (need_swap_orientation) {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (nb_node - k + min_node_index + incr) % nb_node;
      Int32 index2 = ((2 * nb_node - 1) + incr + min_node_index - k) % nb_node;
      sorted_nodes_uid[k] = nodes_uid[index];
      sorted_nodes_uid[k + nb_node] = nodes_uid[index2 + nb_node];
    }
  }
  else {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (k + min_node_index + incr2) % nb_node;
      sorted_nodes_uid[k] = nodes_uid[index];
      sorted_nodes_uid[k + nb_node] = nodes_uid[index + nb_node];
    }
  }
  return need_swap_orientation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NodesOfItemReorderer::
reorder(ItemTypeId type_id, ConstArrayView<Int64> nodes_uids)
{
  ItemTypeInfo* iti = m_item_type_mng->typeFromId(type_id);
  Int32 order = iti->order();
  Int32 nb_node = nodes_uids.size();
  m_work_sorted_nodes.resize(nb_node);
  if (order > 3)
    ARCANE_THROW(NotImplementedException, "Node reordering for type of order 4 or more");
  if (order == 3)
    return _reorderOrder3(nodes_uids, m_work_sorted_nodes, iti->hasCenterNode());
  if (order == 2)
    return _reorderOrder2(nodes_uids, m_work_sorted_nodes, iti->hasCenterNode());
  return MeshUtils::reorderNodesOfFace(nodes_uids, m_work_sorted_nodes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
