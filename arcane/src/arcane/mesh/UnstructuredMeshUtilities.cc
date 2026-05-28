// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshUtilities.cc                                (C) 2000-2025 */
/*                                                                           */
/* Utility functions for a mesh.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/InvalidArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IMeshChecker.h"
#include "arcane/core/IItemFamilyNetwork.h"
#include "arcane/core/NodesOfItemReorderer.h"

#include "arcane/mesh/UnstructuredMeshUtilities.h"
#include "arcane/ConnectivityItemVector.h"
#include "arcane/mesh/NewItemOwnerBuilder.h"
#include "arcane/mesh/ParticleFamily.h"
#include "arcane/mesh/GraphDoFs.h"
#include "arcane/mesh/BasicItemPairGroupComputeFunctor.h"
#include "arcane/mesh/MeshNodeMerger.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"
#include "arcane/mesh/ItemsOwnerBuilder.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnstructuredMeshUtilities::
UnstructuredMeshUtilities(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_compute_adjacency_functor(new BasicItemPairGroupComputeFunctor(traceMng()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnstructuredMeshUtilities::
~UnstructuredMeshUtilities()
{
  delete m_compute_adjacency_functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
changeOwnersFromCells()
{
  // We assume that we know the new owners of the cells, which
  // are found in cells_owner. We must
  // now determine the new owners of the nodes and
  // faces. Pending an algorithm that better balances
  // messages, we apply the following:
  // - each subdomain is responsible for determining the new
  // owner of the nodes and faces belonging to it.
  // - for nodes, the new owner is the new owner
  // of the cell connected to this node whose uniqueId() is the smallest.
  // - for faces, the new owner is the new owner
  // of the cell behind this face if it is an internal face
  // and of the connected cell if it is a boundary face.
  // - for dual nodes, the new owner is the new owner
  // of the cell connected to the dual element
  // - for links, the new owner is the new owner
  // of the cell connected to the first dual node, that is, the owner
  // of the link's first dual node

  // Utility for assigning owners to items
  mesh::NewItemOwnerBuilder owner_builder;

  VariableItemInt32& nodes_owner(m_mesh->nodeFamily()->itemsNewOwner());
  VariableItemInt32& edges_owner(m_mesh->edgeFamily()->itemsNewOwner());
  VariableItemInt32& faces_owner(m_mesh->faceFamily()->itemsNewOwner());
  VariableItemInt32& cells_owner(m_mesh->cellFamily()->itemsNewOwner());

  // Determine the new owners of nodes
  {
    ENUMERATE_NODE (i_node, m_mesh->ownNodes()) {
      const Node node = *i_node;
      const Cell cell = owner_builder.connectedCellOfItem(node);
#ifdef ARCANE_DEBUG_LOAD_BALANCING
      if (nodes_owner[node] != cells_owner[cell]) {
        info() << "New owner for node: " << ItemPrinter(node) << " cell=" << ItemPrinter(cell)
               << " old_owner=" << nodes_owner[node]
               << " current_cell_owner=" << cell.owner()
               << " new_owner=" << cells_owner[cell];
      }
#endif /* ARCANE_DEBUG_LOAD_BALANCING */
      nodes_owner[node] = cells_owner[cell];
    }
    nodes_owner.synchronize();
  }

  // Determine the new owners of edges
  {
    ENUMERATE_EDGE (i_edge, m_mesh->ownEdges()) {
      const Edge edge = *i_edge;
      const Cell cell = owner_builder.connectedCellOfItem(edge);
#ifdef ARCANE_DEBUG_LOAD_BALANCING
      if (edges_owner[edge] != cells_owner[cell]) {
        info() << "New owner for edge: " << ItemPrinter(edge) << " cell=" << ItemPrinter(cell)
               << " old_owner=" << edges_owner[edge]
               << " current_cell_owner=" << cell.owner()
               << " new_owner=" << cells_owner[cell];
      }
#endif /* ARCANE_DEBUG_LOAD_BALANCING */
      edges_owner[edge] = cells_owner[cell];
    }
    edges_owner.synchronize();
  }

  // Determine the new owners of faces
  {
    ENUMERATE_FACE (i_face, m_mesh->ownFaces()) {
      const Face face = *i_face;
      const Cell cell = owner_builder.connectedCellOfItem(face);
      faces_owner[face] = cells_owner[cell];
    }
    faces_owner.synchronize();
  }

  // Determine the new owners of particles
  // Particles have the same owner as the cell they are in.
  for (IItemFamily* family : m_mesh->itemFamilies()) {
    if (family->itemKind() != IK_Particle)
      continue;
    // Position the new owners of particles
    VariableItemInt32& particles_owner(family->itemsNewOwner());
    ENUMERATE_PARTICLE (i_particle, family->allItems()) {
      Particle particle = *i_particle;
      particles_owner[particle] = cells_owner[particle.cell()];
    }
  }

  // GraphOnDoF
  if (m_mesh->itemFamilyNetwork()) {
    // Dof with Mesh Item connectivity
    for (IItemFamily* family : m_mesh->itemFamilies()) {
      if (family->itemKind() != IK_DoF || family->name() == mesh::GraphDoFs::linkFamilyName())
        continue;
      VariableItemInt32& dofs_new_owner(family->itemsNewOwner());
      std::array<Arcane::eItemKind, 5> dualitem_kinds = { IK_Cell, IK_Face, IK_Edge, IK_Node, IK_Particle };
      for (auto dualitem_kind : dualitem_kinds) {
        IItemFamily* dualitem_family = dualitem_kind == IK_Particle ? m_mesh->findItemFamily(dualitem_kind, mesh::ParticleFamily::defaultFamilyName(), false) : m_mesh->itemFamily(dualitem_kind);
        if (dualitem_family) {

          VariableItemInt32& dualitems_new_owner(dualitem_family->itemsNewOwner());
          auto connectivity_name = mesh::connectivityName(family, dualitem_family);
          bool is_dof2dual = true;
          auto connectivity = m_mesh->itemFamilyNetwork()->getConnectivity(family, dualitem_family, connectivity_name);
          if (!connectivity) {
            connectivity = m_mesh->itemFamilyNetwork()->getConnectivity(dualitem_family, family, connectivity_name);
            is_dof2dual = false;
          }

          if (connectivity) {
            ConnectivityItemVector accessor(connectivity);
            if (is_dof2dual) {
              ENUMERATE_ITEM (item, family->allItems().own()) {
                auto connected_items = accessor.connectedItems(ItemLocalId(item));
                if (connected_items.size() > 0) {
                  dofs_new_owner[*item] = dualitems_new_owner[connected_items[0]];
                }
              }
            }
            else {
              ENUMERATE_ITEM (item, dualitem_family->allItems()) {
                ENUMERATE_ITEM (connected_item, accessor.connectedItems(ItemLocalId(item))) {
                  dofs_new_owner[*connected_item] = dualitems_new_owner[*item];
                }
              }
            }
          }
        }
      }
    }
    // Dof with DoF connectivity
    IItemFamily* links_family = m_mesh->findItemFamily(IK_DoF, mesh::GraphDoFs::linkFamilyName(), false);
    if (links_family) {
      VariableItemInt32& links_new_owner(links_family->itemsNewOwner());
      IItemFamily* dualnodes_family = m_mesh->findItemFamily(IK_DoF, mesh::GraphDoFs::dualNodeFamilyName(), false);
      if (dualnodes_family) {
        VariableItemInt32& dualnodes_new_owner(dualnodes_family->itemsNewOwner());
        auto connectivity_name = mesh::connectivityName(links_family, dualnodes_family);
        auto connectivity = m_mesh->itemFamilyNetwork()->getConnectivity(links_family, dualnodes_family, connectivity_name);
        if (connectivity) {
          ConnectivityItemVector accessor(connectivity);
          ENUMERATE_ITEM (item, links_family->allItems().own()) {
            auto connected_items = accessor.connectedItems(ItemLocalId(item));
            if (connected_items.size() > 0) {
              links_new_owner[*item] = dualnodes_new_owner[connected_items[0]];
            }
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
localIdsFromConnectivity(eItemKind item_kind,
                         IntegerConstArrayView items_nb_node,
                         Int64ConstArrayView items_connectivity,
                         Int32ArrayView local_ids,
                         bool allow_null)
{
  Integer nb_item = items_nb_node.size();
  if (item_kind != IK_Face && item_kind != IK_Cell)
    throw InvalidArgumentException(A_FUNCINFO, "item_kind",
                                   "IK_Cell or IK_Face expected",
                                   (int)item_kind);
  if (item_kind == IK_Cell)
    throw NotImplementedException(A_FUNCINFO, "not implemented for cells");

  if (nb_item != local_ids.size())
    throw InvalidArgumentException(A_FUNCINFO, "local_ids",
                                   "Size different from 'items_nb_node'",
                                   (int)item_kind);

  Integer item_connectivity_index = 0;
  Int64UniqueArray buf;
  buf.reserve(256);
  ItemInternalList nodes(m_mesh->itemsInternal(IK_Node));
  for (Integer i = 0; i < nb_item; ++i) {
    Integer current_nb_node = items_nb_node[i];
    Int64ConstArrayView current_nodes(current_nb_node, items_connectivity.data() + item_connectivity_index);
    item_connectivity_index += current_nb_node;
    if (item_kind == IK_Face) {
      buf.resize(current_nb_node);
      mesh_utils::reorderNodesOfFace(current_nodes, buf);
      Int64 first_node_uid = buf[0];
      Int64ArrayView first_node_uid_array(1, &first_node_uid);
      Int32 first_node_lid = CheckedConvert::toInt32(buf[0]);
      Int32ArrayView first_node_lid_array(1, &first_node_lid);
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(first_node_lid_array, first_node_uid_array, !allow_null);
      if (first_node_lid == NULL_ITEM_LOCAL_ID) {
        if (allow_null) {
          local_ids[i] = NULL_ITEM_LOCAL_ID;
        }
        else {
          StringBuilder sb("Face with nodes (");
          for (Integer j = 0; j < current_nb_node; ++j) {
            if (j != 0)
              sb += " ";
            sb += current_nodes[j];
          }
          sb += ") not found (first node ";
          sb += first_node_uid;
          sb += " not found)";
          ARCANE_FATAL(sb.toString());
        }
        continue;
      }

      Node node(nodes[first_node_lid]);
      Face face(mesh_utils::getFaceFromNodesUnique(node, buf));
      if (face.null()) {
        if (allow_null) {
          local_ids[i] = NULL_ITEM_LOCAL_ID;
        }
        else {
          StringBuilder sb("Face with nodes (");
          for (Integer j = 0; j < current_nb_node; ++j) {
            if (j != 0)
              sb += " ";
            sb += current_nodes[j];
          }
          sb += ") not found";
          ARCANE_FATAL(sb.toString());
        }
      }
      else
        local_ids[i] = face.localId();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
getFacesLocalIdFromConnectivity(ConstArrayView<ItemTypeId> items_type,
                                ConstArrayView<Int64> items_connectivity,
                                ArrayView<Int32> local_ids,
                                bool allow_null)
{
  Int32 nb_item = items_type.size();

  if (nb_item != local_ids.size())
    throw InvalidArgumentException(A_FUNCINFO, "local_ids",
                                   "Size different from 'items_type'");

  Int32 item_connectivity_index = 0;
  ItemTypeMng* item_type_mng = m_mesh->itemTypeMng();
  NodesOfItemReorderer face_reorderer(item_type_mng);

  ItemInternalList nodes(m_mesh->itemsInternal(IK_Node));
  for (Integer i = 0; i < nb_item; ++i) {
    ItemTypeId current_type_id = items_type[i];
    Int32 current_nb_node = item_type_mng->typeFromId(current_type_id)->nbLocalNode();
    Int64ConstArrayView current_nodes(current_nb_node, items_connectivity.data() + item_connectivity_index);
    item_connectivity_index += current_nb_node;

    face_reorderer.reorder(current_type_id, current_nodes);
    ConstArrayView<Int64> buf(face_reorderer.sortedNodes());
    Int64 first_node_uid = buf[0];
    Int64ArrayView first_node_uid_array(1, &first_node_uid);
    Int32 first_node_lid = CheckedConvert::toInt32(buf[0]);
    Int32ArrayView first_node_lid_array(1, &first_node_lid);
    m_mesh->nodeFamily()->itemsUniqueIdToLocalId(first_node_lid_array, first_node_uid_array, !allow_null);
    if (first_node_lid == NULL_ITEM_LOCAL_ID) {
      if (allow_null) {
        local_ids[i] = NULL_ITEM_LOCAL_ID;
      }
      else {
        StringBuilder sb("Face with nodes (");
        for (Integer j = 0; j < current_nb_node; ++j) {
          if (j != 0)
            sb += " ";
          sb += current_nodes[j];
        }
        sb += ") not found (first node ";
        sb += first_node_uid;
        sb += " not found)";
        ARCANE_FATAL(sb.toString());
      }
      continue;
    }

    Node node(nodes[first_node_lid]);
    Face face(MeshUtils::getFaceFromNodesUnique(node, buf));
    if (face.null()) {
      if (allow_null) {
        local_ids[i] = NULL_ITEM_LOCAL_ID;
      }
      else {
        StringBuilder sb("Face with nodes (");
        for (Integer j = 0; j < current_nb_node; ++j) {
          if (j != 0)
            sb += " ";
          sb += current_nodes[j];
        }
        sb += ") not found";
        ARCANE_FATAL(sb.toString());
      }
    }
    else
      local_ids[i] = face.localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 UnstructuredMeshUtilities::
_round(Real3 value)
{
  Real3 rvalue = value;
  // Avoid numerical rounding issues
  if (math::isNearlyZero(value.x))
    rvalue.x = 0.0;
  if (math::isNearlyZero(value.y))
    rvalue.y = 0.0;
  if (math::isNearlyZero(value.z))
    rvalue.z = 0.0;

  if (math::isNearlyEqual(value.x, 1.0))
    rvalue.x = 1.0;
  if (math::isNearlyEqual(value.y, 1.0))
    rvalue.y = 1.0;
  if (math::isNearlyEqual(value.z, 1.0))
    rvalue.z = 1.0;

  if (math::isNearlyEqual(value.x, -1.0))
    rvalue.x = -1.0;
  if (math::isNearlyEqual(value.y, -1.0))
    rvalue.y = -1.0;
  if (math::isNearlyEqual(value.z, -1.0))
    rvalue.z = -1.0;

  return rvalue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 UnstructuredMeshUtilities::
computeNormal(const FaceGroup& face_group, const VariableNodeReal3& nodes_coord)
{
  String func_name = "UnstructuredMeshUtilities::computeNormal";
  IParallelMng* pm = m_mesh->parallelMng();
  Integer rank = pm->commRank();
  //Integer nb_item = face_group.size();
  Integer nb_node = 0;
  Integer mesh_dimension = m_mesh->dimension();
  // 'global_normal' and 'total_global_normal' are used to determine
  // a direction for the surface assuming it is
  // an external surface. The direction is then used to orient
  // the calculated normal in the outward normal direction.
  Real3 global_normal = Real3::null();
  Real nb_global_normal_used_node = 0.0;
  ENUMERATE_FACE (iface, face_group) {
    Face face = *iface;
    Integer face_nb_node = face.nbNode();
    nb_node += face_nb_node;

    if (face.isSubDomainBoundary() && face.isOwn()) {
      Real3 face_normal;
      if (mesh_dimension == 3) {
        Real3 v1 = nodes_coord[face.node(1)] - nodes_coord[face.node(0)];
        Real3 v2 = nodes_coord[face.node(2)] - nodes_coord[face.node(0)];
        face_normal = math::vecMul(v1, v2);
      }
      else if (mesh_dimension == 2) {
        Real3 dir = nodes_coord[face.node(0)] - nodes_coord[face.node(1)];
        face_normal.x = -dir.y;
        face_normal.y = dir.x;
        face_normal.z = 0.0;
      }
      if (face.boundaryCell() == face.frontCell())
        face_normal = -face_normal;
      //info() << "ADD NORMAL normal=" << face_normal;
      global_normal = global_normal + face_normal;
      nb_global_normal_used_node += 1.0;
    }
  }
  // The algorithm attempts to determine the nodes at the ends of the surface
  // It assumes these are nodes that belong to only one
  // face of the surface. This works well if all faces have at least
  // four edges. If there are triangles, we assume this is not
  // everywhere, and in this case, we have at least two known extreme nodes.
  typedef HashTableMapT<Int32, Int32> NodeOccurenceMap;
  // Range in the table for each node the number of surface faces
  // to which it is connected.
  NodeOccurenceMap nodes_occurence(nb_node * 2, true, nb_node);
  ENUMERATE_FACE (iface, face_group) {
    Face face = *iface;
    for (NodeLocalId inode : face.nodeIds()) {
      NodeOccurenceMap::Data* v = nodes_occurence.lookup(inode);
      if (v)
        ++v->value();
      else {
        //info() << " ADD NODE " << ItemPrinter(*inode) << " coord=" << nodes_coord[inode];
        nodes_occurence.add(inode, 1);
      }
    }
  }
  UniqueArray<Node> single_nodes;
  NodeLocalIdToNodeConverter nodes_internal(m_mesh->nodeFamily());
  nodes_occurence.each([&](NodeOccurenceMap::Data* d) {
    if (d->value() == 1) {
      Node node = nodes_internal[d->key()];
      // In parallel, only process the nodes that belong to us
      if (node.owner() == rank) {
        single_nodes.add(node);
        //info() << "SINGLE NODE OWNER lid=" << d->key() << " " << ItemPrinter(node)
        //     << " coord=" << nodes_coord[node];
      }
    }
  });
  // Each sub-domain collects the coordinates of the nodes from the other sub-domains
  Integer nb_single_node = single_nodes.size();
  //info() << "NB SINGLE NODE= " << nb_single_node;
  Integer total_nb_single_node = pm->reduce(Parallel::ReduceSum, nb_single_node);
  Real3 total_global_normal;
  {
    Real all_min = 0.0;
    Real all_max = 0.0;
    Real all_sum = 0.0;
    Int32 min_rank = -1;
    Int32 max_rank = -1;
    pm->computeMinMaxSum(nb_global_normal_used_node, all_min, all_max, all_sum, min_rank, max_rank);
    Real3 buf;
    if (max_rank == rank) {
      // I am the one with the farthest point. I send it to the others.
      buf = global_normal;
    }
    pm->broadcast(Real3ArrayView(1, &buf), max_rank);
    total_global_normal = buf;
  }

  info() << "TOTAL SINGLE NODE= " << total_nb_single_node << " surface=" << face_group.name()
         << " global_normal = " << total_global_normal;
  if (total_nb_single_node < 2) {
    ARCANE_FATAL("not enough nodes connected to only one face");
  }
  Real3UniqueArray coords(nb_single_node);
  for (Integer i = 0; i < nb_single_node; ++i) {
    coords[i] = nodes_coord[single_nodes[i]];
  }
  //Array<Real> all_nodes_coord_real;
  UniqueArray<Real3> all_nodes_coord;
  pm->allGatherVariable(coords, all_nodes_coord);
  //info() << " ALL NODES COORD n=" << all_nodes_coord_real.size();
  //Array<Real3> all_nodes_coord(total_nb_single_node);
  Real3 barycentre = Real3::null();
  for (Integer i = 0; i < total_nb_single_node; ++i) {
    barycentre += all_nodes_coord[i];
  }
  barycentre /= (Real)total_nb_single_node;
  if (total_nb_single_node == 2 && m_mesh->dimension() == 3) {
    // We only have two nodes. We need at least a third one to determine
    // the normal to the plane. For this, each processor searches for the node
    // on the surface that is farthest from the barycenter of the two nodes already
    // found. This farthest node will serve as the third point for the
    // plane.
    Real max_distance = 0.0;
    Node farthest_node;
    Real3 s0 = all_nodes_coord[0];
    Real3 s1 = all_nodes_coord[1];
    nodes_occurence.each([&](NodeOccurenceMap::Data* d) {
      Node node = nodes_internal[d->key()];
      Real3 coord = nodes_coord[node];
      // Do not process the two nodes already found
      if (math::isNearlyEqual(coord, s0))
        return;
      if (math::isNearlyEqual(coord, s1))
        return;
      Real distance = (coord - barycentre).squareNormL2();
      if (distance > max_distance) {
        Real3 normal = math::cross(coord - s0, s1 - s0);
        // We only take the node if it is not aligned with the other two.
        if (!math::isNearlyZero(normal.squareNormL2())) {
          max_distance = distance;
          farthest_node = node;
        }
      }
    });

    if (!farthest_node.null()) {
      info() << " FARTHEST NODE= " << ItemPrinter(farthest_node) << " dist=" << max_distance;
    }
    {
      Real3 farthest_coord = _broadcastFarthestNode(max_distance, farthest_node, nodes_coord);
      info() << " FARTHEST NODE ALL coord=" << farthest_coord;
      all_nodes_coord.add(farthest_coord);
    }
  }
  // Sort the nodes so that the calculation does not depend on the order of
  // parallel operations.
  std::sort(std::begin(all_nodes_coord), std::end(all_nodes_coord));
  Integer nb_final_node = all_nodes_coord.size();
  Real3 full_normal = Real3::null();
  info() << " NB FINAL NODE=" << nb_final_node;
  for (Integer i = 0; i < nb_final_node; ++i)
    info() << " NODE=" << all_nodes_coord[i];
  if (m_mesh->dimension() == 2) {
    if (nb_final_node != 2) {
      ARCANE_FATAL("should have 2 border nodes in 2D mesh");
    }
    Real3 direction = all_nodes_coord[1] - all_nodes_coord[0];
    full_normal.x = -direction.y;
    full_normal.y = direction.x;
  }
  else if (m_mesh->dimension() == 3) {
    //nb_final_node = 3; // We only take the first 3 points.
    // NOTE: we could take all points, because if the first three are
    // aligned, the normal will not be correct.
    // If we take all points, we must ensure they are ordered
    // such that the normal of three consecutive points is always
    // in the same direction. This means if we have four points, for example,
    // that these 4 points form a non-crossed quadrangle (not in the shape
    // of a butterfly)
    Real3 first_normal = math::vecMul(all_nodes_coord[2] - all_nodes_coord[0],
                                      all_nodes_coord[1] - all_nodes_coord[0]);
    for (Integer i = 0; i < nb_final_node; ++i) {
      Real3 s0 = all_nodes_coord[i];
      Real3 s1 = all_nodes_coord[(i + nb_final_node - 1) % nb_final_node];
      Real3 s2 = all_nodes_coord[(i + 1) % nb_final_node];
      Real3 normal = math::vecMul(s2 - s0, s1 - s0);
      info() << " ADD NORMAL: " << normal;
      full_normal += normal;
      info() << " FULL: " << full_normal;
    }
    if (math::isNearlyZero(full_normal.squareNormL2()))
      full_normal = first_normal;
  }
  else
    ARCANE_FATAL("invalid mesh dimension (should be 2 or 3)");
  Real a = full_normal.normL2();
  if (math::isZero(a))
    ARCANE_FATAL("invalid value for normal");
  full_normal /= a;
  Real b = total_global_normal.normL2();
  Real dir = 0.0;
  info() << " Normal is " << full_normal;
  if (!math::isZero(b)) {
    total_global_normal /= b;
    dir = math::scaMul(full_normal, total_global_normal);
    if (dir < 0.0)
      full_normal = -full_normal;
  }
  full_normal = _round(full_normal);
  info() << " Final normal is " << full_normal << " dir=" << dir;
  return full_normal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * The algorithm used is as follows:
 * 1. calculates the barycenter of the set of nodes.
 * 2. determines the node farthest from this barycenter, denoted n1
 * 3. determines the node farthest from n1, denoted n2.
 * 4. determines the direction from the coordinates of n1 and n2.
 */
Real3 UnstructuredMeshUtilities::
computeDirection(const NodeGroup& node_group,
                 const VariableNodeReal3& nodes_coord,
                 Real3* n1, Real3* n2)
{
  String func_name = "UnstructuredMeshUtilities::computeDirection";
  IParallelMng* pm = m_mesh->parallelMng();
  //Integer rank = pm->commRank();
  //Integer nb_node_own = node_group.own().size();
  //Integer total_nb_node_own = pm->reduce(Parallel::ReduceSum,nb_node_own);
  Real3 barycentre = Real3::null();
  ENUMERATE_NODE (inode, node_group.own()) {
    barycentre += nodes_coord[inode];
  }
  Real r_barycentre[3];
  r_barycentre[0] = barycentre.x;
  r_barycentre[1] = barycentre.y;
  r_barycentre[2] = barycentre.z;
  RealArrayView rav(3, r_barycentre);
  pm->reduce(Parallel::ReduceSum, rav);
  barycentre = Real3(rav[0], rav[1], rav[2]);
  debug() << " BARYCENTRE COMPUTE DIRECTION = " << barycentre;
  pm->barrier();

  // Determines the node farthest from the barycenter
  Real3 first_boundary_coord;
  {
    Real max_distance = 0.0;
    Node farthest_node;
    ENUMERATE_NODE (inode, node_group.own()) {
      Node node = *inode;
      Real3 coord = nodes_coord[node];
      Real distance = (coord - barycentre).squareNormL2();
      if (distance > max_distance) {
        max_distance = distance;
        farthest_node = node;
      }
    }
    first_boundary_coord = _broadcastFarthestNode(max_distance, farthest_node, nodes_coord);
    if (n1)
      *n1 = first_boundary_coord;
  }
  Real3 second_boundary_coord;
  {
    Real max_distance = 0.0;
    Node farthest_node;
    ENUMERATE_NODE (inode, node_group.own()) {
      Node node = *inode;
      Real3 coord = nodes_coord[node];
      Real distance = (coord - first_boundary_coord).squareNormL2();
      if (distance > max_distance) {
        max_distance = distance;
        farthest_node = node;
      }
    }
    second_boundary_coord = _broadcastFarthestNode(max_distance, farthest_node, nodes_coord);
    if (n2)
      *n2 = second_boundary_coord;
  }
  Real3 direction = second_boundary_coord - first_boundary_coord;
  Real norm = direction.normL2();
  if (math::isZero(norm)) {
    ARCANE_FATAL("Direction is null for group '{0}' first_coord={1} second_coord={2}",
                 node_group.name(), first_boundary_coord, second_boundary_coord);
  }
  return direction / norm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 UnstructuredMeshUtilities::
_broadcastFarthestNode(Real distance, const Node& farthest_node,
                       const VariableNodeReal3& nodes_coord)
{
  String func_name = "UnstructuredMeshUtilities::_broadcastFarthestNode()";
  IParallelMng* pm = m_mesh->parallelMng();
  Integer rank = pm->commRank();

  Real all_min_distance = 0.0;
  Real all_max_distance = 0.0;
  Real all_sum_distance = 0.0;
  Int32 min_rank = -1;
  Int32 max_rank = -1;
  pm->computeMinMaxSum(distance, all_min_distance, all_max_distance, all_sum_distance, min_rank, max_rank);
  Real3 buf;
  if (!farthest_node.null())
    debug() << " FARTHEST NODE myself coord=" << nodes_coord[farthest_node]
            << " distance=" << distance
            << " rank=" << max_rank
            << " max_distance=" << all_max_distance;
  else
    debug() << " FARTHEST NODE myself coord=none"
            << " distance=" << distance
            << " rank=" << max_rank
            << " max_distance=" << all_max_distance;

  if (max_rank == rank) {
    if (farthest_node.null())
      ARCANE_FATAL("can not find farthest node");
    // I am the one with the farthest point. I send it to the others.
    buf = nodes_coord[farthest_node];
    debug() << " I AM FARTHEST NODE ALL coord=" << buf;
  }
  pm->broadcast(Real3ArrayView(1, &buf), max_rank);
  Real3 farthest_coord(buf);
  debug() << " FARTHEST NODE ALL coord=" << farthest_coord
          << " rank=" << max_rank
          << " max_distance=" << all_max_distance;
  return farthest_coord;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
computeAdjency(ItemPairGroup adjency_array, eItemKind link_kind, Integer nb_layer)
{
  m_compute_adjacency_functor->computeAdjacency(adjency_array, link_kind, nb_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool UnstructuredMeshUtilities::
writeToFile(const String& file_name, const String& service_name)
{
  ServiceBuilder<IMeshWriter> sb(m_mesh->handle());
  auto mesh_writer = sb.createReference(service_name, SB_AllowNull);

  if (!mesh_writer) {
    UniqueArray<String> available_names;
    sb.getServicesNames(available_names);
    warning() << String::format("The specified service '{0}' to write the mesh is not available."
                                " Valid names are {1}",
                                service_name, available_names);
    return true;
  }
  mesh_writer->writeMeshToFile(m_mesh, file_name);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
partitionAndExchangeMeshWithReplication(IMeshPartitionerBase* partitioner,
                                        bool initial_partition)
{
  IPrimaryMesh* primary_mesh = partitioner->primaryMesh();
  IMesh* mesh = primary_mesh;
  if (mesh != this->m_mesh)
    throw ArgumentException(A_FUNCINFO, "partitioner->mesh() != this->m_mesh");

  IParallelMng* pm = mesh->parallelMng();
  ITimeStats* ts = pm->timeStats();
  IParallelReplication* pr = pm->replication();
  bool has_replication = pr->hasReplication();

  // In replication mode, only the first replica performs the balancing.
  // It must then send this mesh information to the others
  // so that they have the same mesh.

  info() << "Partition start date=" << platform::getCurrentDateTime();
  if (pr->isMasterRank()) {
    Timer::Action ts_action1(ts, "MeshPartition", true);
    info() << "Partitioning the mesh (initial?=" << initial_partition << ")";
    partitioner->partitionMesh(initial_partition);
  }
  else
    info() << "Waiting for partition information from the master replica";

  if (has_replication) {
    pm->barrier();

    // Checks that all families are the same.
    mesh->checker()->checkValidReplication();

    Int32 replica_master_rank = pr->masterReplicationRank();
    IParallelMng* rep_pm = pr->replicaParallelMng();

    // Only the master replica has the correct owners. It must update
    // the others from this. To do this, we synchronize
    // the owners of the meshes and then update the other families
    // from the mesh owners.
    {
      Int32ArrayView owners = mesh->cellFamily()->itemsNewOwner().asArray();
      rep_pm->broadcast(owners, replica_master_rank);
      if (!pr->isMasterRank()) {
        changeOwnersFromCells();
      }
    }
  }
  info() << "Partition end date=" << platform::getCurrentDateTime();
  {
    Timer::Action ts_action2(ts, "MeshExchange", true);
    primary_mesh->exchangeItems();
  }
  info() << "Exchange end date=" << platform::getCurrentDateTime();
  partitioner->notifyEndPartition();

  // It must be compacted to return to the same situation as
  // if we had just read a mesh directly (which performs a prepareForDump()
  // when calling endAllocate()).
  // We do this also in case of replication to avoid potential
  // inconsistencies if later some replicas call this
  // method and others do not.
  // TODO: check if it should also be done in case of repartitioning.
  if (initial_partition || has_replication)
    mesh->prepareForDump();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
mergeNodes(Int32ConstArrayView nodes_local_id,
           Int32ConstArrayView nodes_to_merge_local_id,
           bool allow_non_corresponding_face)
{
  mesh::MeshNodeMerger merger(m_mesh);
  merger.mergeNodes(nodes_local_id, nodes_to_merge_local_id, allow_non_corresponding_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
computeAndSetOwnersForNodes()
{
  mesh::ItemsOwnerBuilder owner_builder(m_mesh);
  owner_builder.computeNodesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
computeAndSetOwnersForEdges()
{
  mesh::ItemsOwnerBuilder owner_builder(m_mesh);
  owner_builder.computeEdgesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
computeAndSetOwnersForFaces()
{
  mesh::ItemsOwnerBuilder owner_builder(m_mesh);
  owner_builder.computeFacesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
  void _recomputeUniqueIds(IItemFamily* family)
  {
    SmallArray<Int64> unique_ids;
    ENUMERATE_ (ItemWithNodes, iitem, family->allItems()) {
      ItemWithNodes item = *iitem;
      Int32 index = 0;
      unique_ids.resize(item.nbNode());
      for (Node node : item.nodes()) {
        unique_ids[index] = node.uniqueId();
        ++index;
      }
      Int64 new_uid = MeshUtils::generateHashUniqueId(unique_ids);
      item.mutableItemBase().setUniqueId(new_uid);
    }
    family->notifyItemsUniqueIdChanged();
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshUtilities::
recomputeItemsUniqueIdFromNodesUniqueId()
{
  IMesh* mesh = m_mesh;
  ARCANE_CHECK_POINTER(mesh);
  ITraceMng* tm = mesh->traceMng();

  tm->info() << "Calling RecomputeItemsUniqueIdFromNodesUniqueId()";
  // First indicate that the nodes have changed to potentially
  // update the orientation of the faces.
  mesh->nodeFamily()->notifyItemsUniqueIdChanged();
  _recomputeUniqueIds(mesh->edgeFamily());
  _recomputeUniqueIds(mesh->faceFamily());
  _recomputeUniqueIds(mesh->cellFamily());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
