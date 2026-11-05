// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshIncrementalBuilder.cc                            (C) 2000-2026 */
/*                                                                           */
/* Incremental mesh construction.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iterator.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/IItemFamilyModifier.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/OneMeshItemAdder.h"
#include "arcane/mesh/GhostLayerBuilder.h"
#include "arcane/mesh/FaceUniqueIdBuilder.h"
#include "arcane/mesh/EdgeUniqueIdBuilder.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/HashTableMap2.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/core/NodesOfItemReorderer.h"
#include "arcane/mesh/GraphDoFs.h"
#include "arcane/mesh/ParticleFamily.h"

#include <set>
#include <map>
#include <algorithm>
#include <utility>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCANE_DEBUG_DYNAMIC_MESH
// #define ARCANE_DEBUG_DYNAMIC_MESH2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshIncrementalBuilder::
DynamicMeshIncrementalBuilder(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_item_type_mng(mesh->itemTypeMng())
, m_has_amr(mesh->isAmrActivated())
, m_one_mesh_item_adder(new OneMeshItemAdder(this))
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_VERBOSE_MESH_BUILDER", true))
    m_verbose = (v.value() != 0);
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_BULK_ADD_CELLS", true))
    m_use_bulk_add_cells = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshIncrementalBuilder::
~DynamicMeshIncrementalBuilder()
{
  delete m_one_mesh_item_adder;
  delete m_face_unique_id_builder;
  delete m_edge_unique_id_builder;
  delete m_ghost_layer_builder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_printCellFaceInfos(ItemInternal* icell, const String& str)
{
  Cell cell(icell);
  Integer nb_num_back_face = 0;
  Integer nb_num_front_face = 0;
  Integer index = 0;
  for (Face face : cell.faces()) {
    String add_msg;
    if (face.backCell() == cell) {
      ++nb_num_back_face;
      add_msg = "is a back face";
    }
    else {
      ++nb_num_front_face;
      add_msg = "is a front face";
    }
    info() << str << ": celluid=" << cell.uniqueId()
           << " faceuid=" << face.uniqueId()
           << " faceindex=" << index
           << add_msg;
    ++index;
  }
  info() << str << ": celluid=" << cell.uniqueId()
         << " nbback=" << nb_num_back_face
         << " nbfront=" << nb_num_front_face;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds cells to the current mesh.
 *
 * \param mesh_nb_cell number of cells to add
 * \param cells_infos mesh information (see IMesh::allocateMesh())
 * \param sub_domain_id sub-domain to which the cells will belong
 * \param cells returns, if not empty, contains the created cells.
 */
void DynamicMeshIncrementalBuilder::
addCells(Integer nb_cell, Int64ConstArrayView cells_infos,
         Int32 owner, Int32ArrayView cells,
         bool allow_build_face)
{
  // With ARCANE_BULK_ADD_CELLS=1, addCellsBulk() builds the same mesh by filling
  // the family arrays instead of walking item by item. It returns false, having
  // touched nothing, on a mesh it does not build, and the loop below then does
  // the work.
  if (m_use_bulk_add_cells && addCellsBulk(nb_cell, cells_infos, owner, cells, allow_build_face))
    return;
  ItemTypeMng* itm = m_item_type_mng;

  Integer cells_infos_index = 0;
  bool add_to_cells = cells.size() != 0;
  if (add_to_cells && nb_cell != cells.size())
    ARCANE_THROW(ArgumentException, "return array 'cells' has to have same size as number of cells");
  auto* adder = m_one_mesh_item_adder;
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    ItemTypeId item_type_id(CheckedConvert::toInt16(cells_infos[cells_infos_index]));
    ++cells_infos_index;
    Int64 cell_unique_id = cells_infos[cells_infos_index];
    ++cells_infos_index;

    Int32 current_cell_nb_node = 0;
    if (item_type_id == ITI_GenericPolyhedron) {
      // For poly types, the number of nodes is the first value in the connectivity
      current_cell_nb_node = static_cast<Int32>(cells_infos[cells_infos_index]);
    }
    else {
      ItemTypeInfo* it = itm->typeFromId(item_type_id);
      current_cell_nb_node = it->nbLocalNode();
    }
    ConstArrayView<Int64> current_cell_nodes_uid(current_cell_nb_node, &cells_infos[cells_infos_index]);

    ItemInternal* cell = adder->addOneCell(item_type_id, cell_unique_id, owner,
                                           current_cell_nodes_uid, allow_build_face);

    if (add_to_cells)
      cells[i_cell] = cell->localId();
    cells_infos_index += current_cell_nb_node;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Largest face the bulk path handles, sizing its per-face scratch.
//! Not ItemStaticInfo::MAX_FACE_NODE, which is the geometric bound.
static const Integer MAX_BULK_FACE_NODE = 32;

/*!
 * \brief Adds cells by filling the family arrays, instead of one item at a time.
 *
 * addCells() walks the cells and, for each one, allocates the cell, then every
 * node it does not know, then every edge and every face, connecting each as it
 * goes, which is about thirty calls per cell into the connectivity machinery.
 *
 * Here the same traversal only discovers which nodes, edges and faces exist and
 * which cell is on which side of a face, and writes the answers into flat
 * arrays. The families are then filled in one pass each, three arrays per
 * connectivity and one batch per family, with no per-item call at all.
 *
 * The traversal is deliberately identical to addCells()' one, in the same
 * order, so that every localId, every uid and every face orientation come out
 * the same: this is the same mesh, built differently. It handles the first
 * allocation of a mesh of dimension 1, 2 or 3 whose cells all have the mesh
 * dimension, with or without edges, of any order; anything else is left to
 * addCells().
 */
bool DynamicMeshIncrementalBuilder::
addCellsBulk(Integer nb_cell, Int64ConstArrayView cells_infos,
             Integer sub_domain_id, Int32ArrayView cells,
             bool allow_build_face)
{
  DynamicMesh* mesh = m_mesh;
  ItemTypeMng* itm = m_item_type_mng;
  const bool add_to_cells = cells.size() != 0;
  if (add_to_cells && nb_cell != cells.size())
    ARCANE_THROW(ArgumentException, "return array 'cells' has to have same size as number of cells");

  // What this path does not know how to build. Anything here falls back.
  if (nb_cell <= 0 || !allow_build_face)
    return false;
  if (mesh->parentMesh())
    return false;
  if (!mesh->meshKind().isMonoDimension())
    return false;
  const Int32 dimension = mesh->dimension();
  if (dimension < 1 || dimension > 3)
    return false;
  NodeFamily& node_family = mesh->trueNodeFamily();
  EdgeFamily& edge_family = mesh->trueEdgeFamily();
  FaceFamily& face_family = mesh->trueFaceFamily();
  CellFamily& cell_family = mesh->trueCellFamily();
  // With edges, addCells() always connects a cell to its edges and an edge to
  // its nodes, and the other edge connectivities only when the mesh keeps them.
  const bool has_edge = hasEdge();
  auto has_connectivity = [&](Integer c) { return Connectivity::hasConnectivity(m_connectivity, c); };
  const bool has_node_edge = has_edge && has_connectivity(Connectivity::CT_NodeToEdge);
  const bool has_edge_cell = has_edge && has_connectivity(Connectivity::CT_EdgeToCell);
  const bool has_face_edge = has_edge && has_connectivity(Connectivity::CT_FaceToEdge);
  const bool has_edge_face = has_edge && has_connectivity(Connectivity::CT_EdgeToFace);
  if (has_edge) {
    if (!has_connectivity(Connectivity::CT_CellToEdge) || !has_connectivity(Connectivity::CT_EdgeToNode))
      return false;
    if (!edge_family.canAllocateBulk())
      return false;
  }
  // Only a fresh allocation: the localIds handed out are consecutive from 0,
  // and an item already present would have to be looked up, not created. A
  // family that was deallocated reports no item but keeps its ItemInternal
  // slots on a free list, so it is not fresh either.
  if (!node_family.canAllocateBulk() || !face_family.canAllocateBulk() || !cell_family.canAllocateBulk())
    return false;

  // ------------------------------------------------------------- pre-scan
  // Sizes every array of the discovery pass, so that not one of them grows while
  // it is being filled.
  UniqueArray<Int64> cell_uids(nb_cell);
  UniqueArray<Int16> cell_type_ids(nb_cell);
  UniqueArray<Int32> cell_node_index(nb_cell + 1);
  UniqueArray<Int32> cell_face_index(nb_cell + 1);
  UniqueArray<Int32> cell_edge_index(nb_cell + 1);
  Int32 total_cell_node = 0;
  Int32 total_cell_face = 0;
  Int32 total_cell_edge = 0;
  Int32 total_face_edge = 0;
  Int32 max_face_node = 0;
  bool has_high_order_face = false;
  {
    Integer k = 0;
    for (Integer i = 0; i < nb_cell; ++i) {
      ItemTypeId type_id{ static_cast<Int16>(cells_infos[k]) };
      // A generic polyhedron carries its own node count in cells_infos and
      // its faces elsewhere: addCells() builds it.
      if (type_id == ITI_GenericPolyhedron)
        return false;
      ItemTypeInfo* type = itm->typeFromId(type_id);
      const Integer nb_node = type->nbLocalNode();
      const Integer nb_face = type->nbLocalFace();
      cell_type_ids[i] = type_id;
      cell_uids[i] = cells_infos[k + 1];
      cell_node_index[i] = total_cell_node;
      cell_face_index[i] = total_cell_face;
      cell_edge_index[i] = total_cell_edge;
      total_cell_node += nb_node;
      total_cell_face += nb_face;
      if (has_edge)
        total_cell_edge += type->nbLocalEdge();
      for (Integer f = 0; f < nb_face; ++f) {
        const Integer n = type->localFace(f).nbNode();
        if (n > MAX_BULK_FACE_NODE)
          return false; // the per-face scratch below is fixed-size
        if (n > max_face_node)
          max_face_node = n;
        if (has_edge)
          total_face_edge += type->localFace(f).nbEdge();
        // A face of order 2 or more is connected to all its nodes, but only its
        // linear nodes know it (see the node to face fill below).
        ItemTypeInfo* face_type = itm->typeFromId(ItemTypeId::fromInteger(type->localFace(f).typeId()));
        if (face_type->linearTypeInfo()->nbLocalNode() != n)
          has_high_order_face = true;
      }
      k += 2 + nb_node;
      if (k > cells_infos.size())
        return false;
    }
    cell_node_index[nb_cell] = total_cell_node;
    cell_face_index[nb_cell] = total_cell_face;
    cell_edge_index[nb_cell] = total_cell_edge;
  }

  // ---------------------------------------------------------------- phase A
  // Discovery. One pass over cells_infos, in the order addCells() uses, so that
  // every localId, every uid and every face orientation come out the same.

  UniqueArray<Int32> cell_nodes(total_cell_node);
  UniqueArray<Int32> cell_faces(total_cell_face);
  // A cell face creates at most one face, so these bound the face arrays.
  UniqueArray<Int64> face_uids(total_cell_face);
  UniqueArray<Int16> face_type_ids(total_cell_face);
  UniqueArray<Int32> face_node_index(total_cell_face + 1);
  UniqueArray<Int32> face_nodes(static_cast<Int64>(total_cell_face) * max_face_node);
  // With faces of order 2 or more, the linear nodes of each face apart: they
  // are the first of its sorted nodes.
  UniqueArray<Int32> face_linear_index(has_high_order_face ? total_cell_face + 1 : 0);
  UniqueArray<Int32> face_linear_nodes(has_high_order_face ? static_cast<Int64>(total_cell_face) * max_face_node : 0);
  Int32 nb_face_linear_node = 0;
  if (has_high_order_face)
    face_linear_index[0] = 0;
  UniqueArray<Int32> face_back_cell(total_cell_face);
  UniqueArray<Int32> face_front_cell(total_cell_face);
  UniqueArray<Int64> node_uids(total_cell_node);
  Int32 nb_node = 0;
  Int32 nb_face = 0;
  Int32 nb_face_node = 0;
  face_node_index[0] = 0;

  impl::HashTableMap2<Int64, Int32> node_map(total_cell_node / 4 + 16, true);
  /*
   * A face is found by walking the faces already attached to its first sorted
   * node: they are a handful, and they were created within a few cells of each
   * other, so the walk stays in cache. This is what findFaceInNode2() does, over
   * arrays this pass owns.
   */
  UniqueArray<Int32> node_first_face(total_cell_node);
  node_first_face.fill(-1);
  UniqueArray<Int32> face_chain_next(total_cell_face);

  // The edges, when the mesh has some. An edge is its two nodes, the one of
  // smaller uniqueId first, and is found among the edges attached to that node,
  // as findEdgeInNode2() finds it. A cell edge creates at most one edge.
  UniqueArray<Int32> cell_edges(total_cell_edge);
  UniqueArray<Int64> edge_uids(total_cell_edge);
  UniqueArray<Int32> edge_node_index(total_cell_edge + 1);
  UniqueArray<Int32> edge_nodes(2 * static_cast<Int64>(total_cell_edge));
  UniqueArray<Int32> node_first_edge(has_edge ? total_cell_node : 0);
  node_first_edge.fill(-1);
  UniqueArray<Int32> edge_chain_next(total_cell_edge);
  // A face's edges are connected when the face is created, so they follow the
  // face order, as the faces' nodes do.
  UniqueArray<Int32> face_edge_index(has_edge ? total_cell_face + 1 : 0);
  UniqueArray<Int32> face_edges(total_face_edge);
  Int32 nb_edge = 0;
  Int32 nb_face_edge = 0;
  if (has_edge)
    face_edge_index[0] = 0;
  Int64 next_edge_uid = m_one_mesh_item_adder->nextEdgeUid();

  NodesOfItemReorderer reorderer(itm);
  SmallArray<Int64, 16> work_face_nodes;
  FixedArray<Int32, MAX_BULK_FACE_NODE> work_face_lids;
  FixedArray<Int32, MAX_BULK_FACE_NODE> sorted_lids;
  Int64 next_face_uid = m_one_mesh_item_adder->nextFaceUid();
  // As in addCells(), a face's uniqueId is either the next one or a hash of its
  // sorted node uniqueIds.
  const bool use_hash = m_one_mesh_item_adder->isUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId();
  FixedArray<Int64, MAX_BULK_FACE_NODE> sorted_uids;

  Integer infos_index = 0;
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    ItemTypeId cell_type_id{ static_cast<Int16>(cells_infos[infos_index]) };
    ItemTypeInfo* cell_type = itm->typeFromId(cell_type_id);
    const Integer cell_nb_node = cell_type->nbLocalNode();
    Int64ConstArrayView cell_nodes_uid(cell_nb_node, &cells_infos[infos_index + 2]);
    infos_index += 2 + cell_nb_node;

    const Int32 node_base = cell_node_index[i_cell];
    for (Integer i = 0; i < cell_nb_node; ++i) {
      auto r = node_map.try_emplace(cell_nodes_uid[i], nb_node);
      if (r.second)
        node_uids[nb_node++] = cell_nodes_uid[i];
      cell_nodes[node_base + i] = r.first->second;
    }

    // addCells() adds a cell's edges before its faces.
    const Int32 edge_base = cell_edge_index[i_cell];
    if (has_edge) {
      const Integer cell_nb_edge = cell_type->nbLocalEdge();
      for (Integer i_edge = 0; i_edge < cell_nb_edge; ++i_edge) {
        const ItemTypeInfo::LocalEdge& le = cell_type->localEdge(i_edge);
        Int32 first = cell_nodes[node_base + le.beginNode()];
        Int32 second = cell_nodes[node_base + le.endNode()];
        if (node_uids[first] > node_uids[second])
          std::swap(first, second);
        Int32 edge_index = -1;
        for (Int32 e = node_first_edge[first]; e >= 0; e = edge_chain_next[e]) {
          if (edge_nodes[2 * e + 1] == second) {
            edge_index = e;
            break;
          }
        }
        if (edge_index < 0) {
          edge_index = nb_edge;
          if (use_hash) {
            FixedArray<Int64, 2> uids;
            uids[0] = node_uids[first];
            uids[1] = node_uids[second];
            edge_uids[nb_edge] = MeshUtils::generateHashUniqueId(uids.view());
          }
          else
            edge_uids[nb_edge] = next_edge_uid++;
          edge_nodes[2 * nb_edge] = first;
          edge_nodes[2 * nb_edge + 1] = second;
          edge_node_index[nb_edge] = 2 * nb_edge;
          edge_chain_next[nb_edge] = node_first_edge[first];
          node_first_edge[first] = nb_edge;
          ++nb_edge;
        }
        cell_edges[edge_base + i_edge] = edge_index;
      }
    }

    // A face is identified by its sorted node list, exactly as findFaceInNode2()
    // identifies it.
    const Integer cell_nb_face = cell_type->nbLocalFace();
    const Int32 face_base = cell_face_index[i_cell];
    for (Integer i_face = 0; i_face < cell_nb_face; ++i_face) {
      const ItemTypeInfo::LocalFace& lf = cell_type->localFace(i_face);
      const Integer face_nb_node = lf.nbNode();
      work_face_nodes.resize(face_nb_node);
      for (Integer z = 0; z < face_nb_node; ++z) {
        const Int32 local = lf.node(z);
        work_face_nodes[z] = cell_nodes_uid[local];
        work_face_lids[z] = cell_nodes[node_base + local];
      }
      bool is_reorder = false;
      if (dimension == 1)
        is_reorder = reorderer.reorder1D(i_face, work_face_nodes[0]);
      else
        is_reorder = reorderer.reorder(ItemTypeId::fromInteger(lf.typeId()), work_face_nodes);
      ConstArrayView<Int64> sorted_nodes = reorderer.sortedNodes();

      // The sorted nodes are a permutation of the face's own nodes, so their
      // localIds are found among the handful just gathered, with no trip through
      // the node map.
      for (Integer z = 0; z < face_nb_node; ++z) {
        const Int64 uid = sorted_nodes[z];
        Int32 lid = NULL_ITEM_LOCAL_ID;
        for (Integer w = 0; w < face_nb_node; ++w) {
          if (work_face_nodes[w] == uid) {
            lid = work_face_lids[w];
            break;
          }
        }
        sorted_lids[z] = lid;
      }

      const Int32 first_node = sorted_lids[0];
      Int32 face_index = -1;
      for (Int32 f = node_first_face[first_node]; f >= 0; f = face_chain_next[f]) {
        const Int32 begin = face_node_index[f];
        if (face_node_index[f + 1] - begin != face_nb_node)
          continue;
        bool same = true;
        for (Integer z = 0; same && z < face_nb_node; ++z)
          same = (face_nodes[begin + z] == sorted_lids[z]);
        if (same) {
          face_index = f;
          break;
        }
      }
      if (face_index < 0) {
        face_index = nb_face;
        if (use_hash) {
          for (Integer z = 0; z < face_nb_node; ++z)
            sorted_uids[z] = node_uids[sorted_lids[z]];
          face_uids[nb_face] = MeshUtils::generateHashUniqueId(SmallSpan<const Int64>(sorted_uids.data(), face_nb_node));
        }
        else
          face_uids[nb_face] = next_face_uid++;
        face_type_ids[nb_face] = static_cast<Int16>(lf.typeId());
        face_back_cell[nb_face] = NULL_ITEM_LOCAL_ID;
        face_front_cell[nb_face] = NULL_ITEM_LOCAL_ID;
        for (Integer z = 0; z < face_nb_node; ++z)
          face_nodes[nb_face_node + z] = sorted_lids[z];
        nb_face_node += face_nb_node;
        face_node_index[nb_face + 1] = nb_face_node;
        if (has_high_order_face) {
          const Integer face_nb_linear_node = itm->typeFromId(ItemTypeId::fromInteger(lf.typeId()))->linearTypeInfo()->nbLocalNode();
          for (Integer z = 0; z < face_nb_linear_node; ++z)
            face_linear_nodes[nb_face_linear_node + z] = sorted_lids[z];
          nb_face_linear_node += face_nb_linear_node;
          face_linear_index[nb_face + 1] = nb_face_linear_node;
        }
        face_chain_next[nb_face] = node_first_face[first_node];
        node_first_face[first_node] = nb_face;
        if (has_edge) {
          const Integer face_nb_edge = lf.nbEdge();
          for (Integer z = 0; z < face_nb_edge; ++z)
            face_edges[nb_face_edge + z] = cell_edges[edge_base + lf.edge(z)];
          nb_face_edge += face_nb_edge;
          face_edge_index[nb_face + 1] = nb_face_edge;
        }
        ++nb_face;
      }
      cell_faces[face_base + i_face] = face_index;
      // is_reorder means this cell is in FRONT of the face.
      Int32& side = is_reorder ? face_front_cell[face_index] : face_back_cell[face_index];
      if (side != NULL_ITEM_LOCAL_ID)
        ARCANE_FATAL("face already has a cell on this side (face index={0})", face_index);
      side = i_cell;
    }
  }

  // ---------------------------------------------------------------- phase B
  // The items themselves, one batch per family. The discovery order is also the
  // local id order, so the arrays above index the families directly.

  UniqueArray<Int32> lids(std::max(std::max(nb_node, nb_edge), std::max(nb_face, static_cast<Int32>(nb_cell))));
  node_family.allocateItemsBulk(node_uids.subConstView(0, nb_node), sub_domain_id,
                                lids.subView(0, nb_node));
  if (has_edge) {
    edge_node_index[nb_edge] = 2 * nb_edge;
    edge_family.allocateItemsBulk(edge_uids.subConstView(0, nb_edge), sub_domain_id,
                                  lids.subView(0, nb_edge));
    for (Int32 i = 0; i < nb_edge; ++i) {
      if (lids[i] != i)
        ARCANE_FATAL("unexpected localId {0} for index {1} in family {2}", lids[i], i, edge_family.name());
    }
  }
  _allocateBulkByType(&face_family, face_uids.subConstView(0, nb_face),
                      face_type_ids.subConstView(0, nb_face), sub_domain_id, lids);
  _allocateBulkByType(&cell_family, cell_uids.constView(), cell_type_ids.constView(),
                      sub_domain_id, lids);

  // ---------------------------------------------------------------- phase C
  // The connectivities. Each is three arrays; the layout has to be the one the
  // incremental path produces, since compactConnectivityList() checks it: the
  // null item holds the first block, and every block starts on a multiple of
  // preAllocatedSize().

  _fillBulkConnectivity(cell_family.nodeConnectivity(), nb_cell, cell_node_index, cell_nodes);
  _fillBulkConnectivity(cell_family.faceConnectivity(), nb_cell, cell_face_index, cell_faces);
  _fillBulkConnectivity(face_family.nodeConnectivity(), nb_face,
                        face_node_index.subConstView(0, nb_face + 1),
                        face_nodes.subConstView(0, nb_face_node));
  // face -> cell is written by FaceFamily, the class allowed to set the
  // II_Boundary / II_HasBackCell / II_HasFrontCell flags that go with it.
  face_family.setBulkBackAndFrontCells(face_back_cell.subConstView(0, nb_face),
                                       face_front_cell.subConstView(0, nb_face));
  _fillBulkReverseConnectivity(node_family.cellConnectivity(), nb_node, nb_cell,
                               cell_node_index, cell_nodes);
  // As in addCells(), a face is added to its linear nodes only, and only when
  // the mesh keeps the node to face connectivity (a 1D mesh does not, by default).
  if (has_connectivity(Connectivity::CT_NodeToFace)) {
    if (has_high_order_face)
      _fillBulkReverseConnectivity(node_family.faceConnectivity(), nb_node, nb_face,
                                   face_linear_index.subConstView(0, nb_face + 1),
                                   face_linear_nodes.subConstView(0, nb_face_linear_node));
    else
      _fillBulkReverseConnectivity(node_family.faceConnectivity(), nb_node, nb_face,
                                   face_node_index.subConstView(0, nb_face + 1),
                                   face_nodes.subConstView(0, nb_face_node));
  }
  if (has_edge) {
    const ConstArrayView<Int32> e_index = edge_node_index.subConstView(0, nb_edge + 1);
    const ConstArrayView<Int32> e_nodes = edge_nodes.subConstView(0, 2 * nb_edge);
    const ConstArrayView<Int32> f_index = face_edge_index.subConstView(0, nb_face + 1);
    const ConstArrayView<Int32> f_edges = face_edges.subConstView(0, nb_face_edge);
    _fillBulkConnectivity(cell_family.edgeConnectivity(), nb_cell, cell_edge_index, cell_edges);
    _fillBulkConnectivity(edge_family.nodeConnectivity(), nb_edge, e_index, e_nodes);
    if (has_node_edge)
      _fillBulkReverseConnectivity(node_family.edgeConnectivity(), nb_node, nb_edge, e_index, e_nodes);
    if (has_edge_cell)
      _fillBulkReverseConnectivity(edge_family.cellConnectivity(), nb_edge, nb_cell, cell_edge_index, cell_edges);
    if (has_face_edge)
      _fillBulkConnectivity(face_family.edgeConnectivity(), nb_face, f_index, f_edges);
    if (has_edge_face)
      _fillBulkReverseConnectivity(edge_family.faceConnectivity(), nb_edge, nb_face, f_index, f_edges);
  }
  if (add_to_cells) {
    for (Integer i = 0; i < nb_cell; ++i)
      cells[i] = i;
  }

  m_one_mesh_item_adder->setNextFaceUid(next_face_uid);
  m_one_mesh_item_adder->setNextEdgeUid(next_edge_uid);
  m_one_mesh_item_adder->addBulkMeshInfos(nb_node, nb_edge, nb_face, nb_cell);

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocates items in batches of one type, since a batch carries one shared info.
 *
 * Checks that the localIds handed out are the position in \a uids, which is
 * what lets the discovery arrays index the families directly.
 */
template <typename FamilyType> void DynamicMeshIncrementalBuilder::
_allocateBulkByType(FamilyType* family, Int64ConstArrayView uids,
                    ConstArrayView<Int16> type_ids, Int32 owner, Int32ArrayView lids)
{
  ItemTypeMng* itm = m_item_type_mng;
  const Int32 nb = uids.size();
  Int32 begin = 0;
  while (begin < nb) {
    Int32 end = begin + 1;
    while (end < nb && type_ids[end] == type_ids[begin])
      ++end;
    ItemTypeInfo* type = itm->typeFromId(ItemTypeId(type_ids[begin]));
    Int32ArrayView batch = lids.subView(begin, end - begin);
    family->allocateItemsBulk(uids.subConstView(begin, end - begin), type, owner, batch);
    begin = end;
  }
  for (Int32 i = 0; i < nb; ++i) {
    if (lids[i] != i)
      ARCANE_FATAL("unexpected localId {0} for index {1} in family {2}",
                   lids[i], i, family->name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fills a connectivity whose blocks are given in source-item order.
 *
 * \a offsets has nb_source+1 entries and \a targets the concatenated blocks.
 */
void DynamicMeshIncrementalBuilder::
_fillBulkConnectivity(IncrementalItemConnectivity* c, Int32 nb_source,
                      ConstArrayView<Int32> offsets, ConstArrayView<Int32> targets)
{
  const Int32 prealloc = c->preAllocatedSize();
  const Int32 head = (prealloc <= 0) ? 1 : prealloc;
  Int64 total = head;
  for (Int32 i = 0; i < nb_source; ++i) {
    const Int32 n = offsets[i + 1] - offsets[i];
    if (n > 0)
      total += (prealloc > 0) ? (prealloc * ((n + prealloc - 1) / prealloc)) : n;
  }
  Int32ArrayView nb_item, index, list;
  c->resizeBulkConnectivity(nb_source, total, nb_item, index, list);
  for (Int32 z = 0; z < head; ++z)
    list[z] = NULL_ITEM_LOCAL_ID;
  Int64 pos = head;
  for (Int32 i = 0; i < nb_source; ++i) {
    const Int32 begin = offsets[i];
    const Int32 n = offsets[i + 1] - begin;
    if (n == 0) {
      index[i] = 0;
      nb_item[i] = 0;
      continue;
    }
    index[i] = CheckedConvert::toInt32(pos);
    nb_item[i] = n;
    for (Int32 z = 0; z < n; ++z)
      list[pos + z] = targets[begin + z];
    const Int32 alloc = (prealloc > 0) ? (prealloc * ((n + prealloc - 1) / prealloc)) : n;
    for (Int32 z = n; z < alloc; ++z)
      list[pos + z] = NULL_ITEM_LOCAL_ID;
    pos += alloc;
  }
  c->endBulkConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fills the reverse of a connectivity given in source-item order.
 *
 * Counting sort: one pass to count how many sources each target has, a prefix
 * sum over the blocks, then one scatter. The sources of a given target come out
 * in increasing source order, which is the order addCells() produces since it
 * connects them as it walks the cells.
 */
void DynamicMeshIncrementalBuilder::
_fillBulkReverseConnectivity(IncrementalItemConnectivity* c, Int32 nb_target,
                             Int32 nb_source, ConstArrayView<Int32> offsets,
                             ConstArrayView<Int32> targets)
{
  const Int32 prealloc = c->preAllocatedSize();
  const Int32 head = (prealloc <= 0) ? 1 : prealloc;
  UniqueArray<Int32> count(nb_target);
  count.fill(0);
  for (Int32 i = 0; i < nb_source; ++i) {
    for (Int32 z = offsets[i]; z < offsets[i + 1]; ++z)
      ++count[targets[z]];
  }

  UniqueArray<Int32> fill_pos(nb_target);
  Int64 total = head;
  for (Int32 t = 0; t < nb_target; ++t) {
    const Int32 n = count[t];
    if (n == 0) {
      fill_pos[t] = 0;
      continue;
    }
    fill_pos[t] = CheckedConvert::toInt32(total);
    total += (prealloc > 0) ? (prealloc * ((n + prealloc - 1) / prealloc)) : n;
  }
  Int32ArrayView nb_item, index, list;
  c->resizeBulkConnectivity(nb_target, total, nb_item, index, list);
  list.fill(NULL_ITEM_LOCAL_ID);
  for (Int32 t = 0; t < nb_target; ++t) {
    nb_item[t] = count[t];
    index[t] = fill_pos[t];
  }
  for (Int32 i = 0; i < nb_source; ++i) {
    for (Int32 z = offsets[i]; z < offsets[i + 1]; ++z) {
      const Int32 t = targets[z];
      list[fill_pos[t]] = i;
      ++fill_pos[t];
    }
  }
  c->endBulkConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds cells to the current mesh. Uses generic item addition based on family dependencies.
 *
 * \param mesh_nb_cell number of cells to add
 * \param cells_infos mesh information (see IMesh::allocateMesh())
 * \param sub_domain_id sub-domain to which the cells will belong
 * \param cells returns, if not empty, contains the created cells.
 */

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addCells3(Integer nb_cell, Int64ConstArrayView cells_infos,
          Integer sub_domain_id, Int32ArrayView cells,
          bool allow_build_face)
{
  bool add_to_cells = cells.size() != 0;
  if (add_to_cells && nb_cell != cells.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'cells' has to have same size as number of cells");
  ItemDataList item_data_list; // store item ids and dependency connectivities
  ItemDataList item_relation_data_list; // store relation connectivities
  ItemData& cell_data = item_data_list.itemData(Integer(m_mesh->cellFamily()->itemKind()),
                                                nb_cell, 0, cells, &mesh()->trueCellFamily(), &mesh()->trueCellFamily(), sub_domain_id);
  // item info_size cannot be guessed
  Int64UniqueArray faces_infos, edges_infos, nodes_infos, node_uids;
  Int32UniqueArray face_lids, edge_lids, node_lids;
  Integer nb_face = 0, nb_edge = 0, nb_node = 0;
  std::map<Int64, Int64SharedArray> cell_to_face_connectivity_info;
  std::map<std::pair<Int64, Int64>, Int64> edge_uid_map;

  // Fill classical info
  _fillFaceInfo(nb_face, nb_cell, faces_infos, cells_infos, cell_to_face_connectivity_info);
  _fillEdgeInfo(nb_edge, nb_face, edges_infos, faces_infos, edge_uid_map);
  _fillNodeInfo(nb_node, nb_face, nodes_infos, faces_infos);

  // Fill Cell info 2
  _fillCellNewInfoNew(nb_cell, cells_infos, cell_data.itemInfos(), cell_to_face_connectivity_info, edge_uid_map);
  // Fill Face info 2
  if (allow_build_face) {
    ItemData& face_data = item_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()), nb_face, 0, face_lids, &mesh()->trueFaceFamily(), &mesh()->trueFaceFamily(), sub_domain_id);
    face_lids.resize(item_data_list[m_mesh->faceFamily()->itemKind()].nbItems());
    _fillFaceNewInfoNew(nb_face, faces_infos, face_data.itemInfos(), edge_uid_map); // todo voir appel depuis add Faces
    ItemData& face_relation_data = item_relation_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()), nb_face, 0, face_lids, &mesh()->trueFaceFamily(), &mesh()->trueFaceFamily(), sub_domain_id);
    _initFaceRelationInfo(face_relation_data, cell_data, faces_infos); // face-cell relations
  }
  // Fill Edge info 2
  if (hasEdge()) {
    Integer edge_info_size = 1 + nb_edge * 6; //New info = Nb_connected_family + for each edge : type, uid, connected_family_kind, nb_node_connected, first_node_uid, second_node_uid
    ItemData& edge_data = item_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()), nb_edge, edge_info_size, edge_lids, &mesh()->trueEdgeFamily(), &mesh()->trueEdgeFamily(), sub_domain_id);
    edge_lids.resize(item_data_list[m_mesh->edgeFamily()->itemKind()].nbItems());
    _fillEdgeNewInfoNew(nb_edge, edges_infos, edge_data.itemInfos()); // reprendre fillEdgeInfo et voir l'appel depuis add Edges + attention ordre des noeuds,
    ItemData& edge_relation_data = item_relation_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()), nb_edge, edge_info_size, edge_lids, &mesh()->trueEdgeFamily(), &mesh()->trueEdgeFamily(), sub_domain_id);
    _initEdgeRelationInfo(edge_relation_data, cell_data, edges_infos); // edge-cell relations
    if (allow_build_face) {
      _appendEdgeRelationInfo(edge_relation_data, item_data_list[IK_Face], edges_infos); // edge-face relations
      _appendFaceRelationInfo(item_relation_data_list[IK_Face], edge_data, faces_infos); // face-edge relations
    }
  }
  // Fill Node info 2
  Integer node_info_size = 1 + nb_node * 2; //New info = Nb_connected_family + for each node : type, uid (no connected family)
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()), nb_node, node_info_size, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  node_lids.resize(item_data_list[m_mesh->nodeFamily()->itemKind()].nbItems());
  _fillNodeNewInfo(nodes_infos, node_data.itemInfos());
  ItemData& node_relation_data = item_relation_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()), nb_node, node_info_size, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  _initNodeRelationInfo(node_relation_data, cell_data, nodes_infos); // node-cell relations
  if (allow_build_face)
    _appendNodeRelationInfo(node_relation_data, item_data_list[IK_Face], nodes_infos); // node-face relations
  if (hasEdge())
    _appendNodeRelationInfo(node_relation_data, item_data_list[IK_Edge], nodes_infos); // node-edge relations

  // Launch node, face and edges (not mandatory) and cell creation
  addItems(item_data_list, item_relation_data_list);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addCells2(Integer nb_cell, Int64ConstArrayView cells_infos,
          Integer sub_domain_id, Int32ArrayView cells,
          bool allow_build_face)
{
  ItemDataList item_data_list;
  ItemData& cell_data = item_data_list.itemData(Integer(m_mesh->cellFamily()->itemKind()),
                                                nb_cell, 0, cells, &mesh()->trueCellFamily(), &mesh()->trueCellFamily(), sub_domain_id);
  // item info_size cannot be guessed
  Int64Array& cell_infos2 = cell_data.itemInfos();
  Int64UniqueArray faces_infos;
  Int64UniqueArray node_uids;
  Int32UniqueArray face_lids;
  Integer nb_face;

  // Fill cell info 2
  _fillCellInfo2(nb_cell, cells_infos, cell_infos2, nb_face, faces_infos, node_uids, allow_build_face);

  // Fill face info 2
  if (allow_build_face) {
    Integer face_infos2_size = faces_infos.size() + 1 + 2 * nb_face; // additive info nb_connected_family + nb_face * (family_id + nb_connected_element)
    face_lids.resize(nb_face);
    ItemData& face_data = item_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),
                                                  nb_face, face_infos2_size, face_lids, &mesh()->trueFaceFamily(), &mesh()->trueFaceFamily(), sub_domain_id);
    Int64ArrayView face_infos2 = face_data.itemInfos().view();
    _fillFaceInfo2(nb_face, faces_infos, face_infos2, node_uids); // if allow_build_face false, node_uids built in fillCellInfo3
  }

  // Fill node info 2
  Integer nb_node_to_add = node_uids.size();
  Int32UniqueArray node_lids(nb_node_to_add);
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
                                                nb_node_to_add, 2 * nb_node_to_add + 1, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  Int64Array& node_infos2 = node_data.itemInfos();
  _fillNodeNewInfo(node_uids, node_infos2);

  // Launch node and face and cell creation
  addItems(item_data_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds cells to the current mesh.
 *
 * \param mesh_nb_cell number of cells to add
 * \param cells_infos mesh information (see IMesh::allocateMesh())
 * \param sub_domain_id sub-domain to which the cells will belong
 * \param cells returns, if not empty, contains the created cells.
 */
void DynamicMeshIncrementalBuilder::
_fillFaceInfo(Integer& nb_face, Integer nb_cell, Int64Array& faces_infos, Int64ConstArrayView cells_infos, std::map<Int64, Int64SharedArray>& cell_to_face_connectivity_info)
{
  faces_infos.reserve(2 * cells_infos.size());
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  ItemTypeMng* itm = m_item_type_mng;
  Integer cells_infos_index = 0;
  nb_face = 0;
  NodeInFaceSet face_nodes_set;
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    Integer item_type_id = (Integer)cells_infos[cells_infos_index++];
    Int64 current_cell_unique_id = cells_infos[cells_infos_index++];
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_cell_nb_node = it->nbLocalNode();
    Int64ConstArrayView current_cell_node_uids(current_cell_nb_node, &cells_infos[cells_infos_index]);
    Integer current_cell_nb_face = it->nbLocalFace();
    for (Integer face_in_cell_index = 0; face_in_cell_index < current_cell_nb_face; ++face_in_cell_index) {
      const ItemTypeInfo::LocalFace& current_face = it->localFace(face_in_cell_index);
      Integer current_face_nb_node = current_face.nbNode();
      work_face_orig_nodes.resize(current_face_nb_node);
      work_face_sorted_nodes.resize(current_face_nb_node);
      Integer i = 0;
      for (Integer node_in_face_index = 0; node_in_face_index < current_face_nb_node; ++node_in_face_index) {
        work_face_orig_nodes[i++] = current_cell_node_uids[current_face.node(node_in_face_index)];
      }
      mesh_utils::reorderNodesOfFace(work_face_orig_nodes, work_face_sorted_nodes);
      // get face uid: we shouldn't have to guess, it should be given in arguments...
      Int64 current_face_uid = _findFaceUniqueId(work_face_sorted_nodes, face_nodes_set);
      if (current_face_uid == NULL_ITEM_ID) // face not already known add to face info
      {
        current_face_uid = m_face_uid_pool++;
        faces_infos.add(current_face.typeId()); // face info type
        faces_infos.add(current_face_uid); // face info uid
        faces_infos.addRange(work_face_sorted_nodes); // face info node uids
        _addFaceNodes(face_nodes_set, work_face_sorted_nodes, current_face_uid);
        ++nb_face;
      }
      cell_to_face_connectivity_info[current_cell_unique_id].add(current_face_uid);
    }
    cells_infos_index += current_cell_nb_node;
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillEdgeInfo(Integer& nb_edge, Integer nb_face, Int64Array& edges_infos,
              Int64ConstArrayView faces_infos, std::map<std::pair<Int64, Int64>, Int64>& edge_uid_map)
{
  edges_infos.reserve(2 * faces_infos.size());
  ItemTypeMng* itm = m_item_type_mng;
  Integer faces_infos_index = 0;
  nb_edge = 0;
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {
    Integer item_type_id = (Integer)faces_infos[faces_infos_index++]; // face type
    ++faces_infos_index; // face uid (unused)
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_face_nb_node = it->nbLocalNode();
    Int64ConstArrayView current_face_node_uids(current_face_nb_node, &faces_infos[faces_infos_index]);
    Integer current_face_nb_edge = it->nbLocalEdge();
    for (Integer edge_in_face_index = 0; edge_in_face_index < current_face_nb_edge; ++edge_in_face_index) {
      const ItemTypeInfo::LocalEdge& current_edge = it->localEdge(edge_in_face_index);
      Int64 first_node = current_face_node_uids[current_edge.beginNode()];
      Int64 second_node = current_face_node_uids[current_edge.endNode()];
      if (first_node > second_node)
        std::swap(first_node, second_node);
      auto edge_it = edge_uid_map.insert(std::make_pair(std::make_pair(first_node, second_node), m_edge_uid_pool));
      if (edge_it.second) {
        edges_infos.add(m_edge_uid_pool); // edge uid
        edges_infos.add(first_node); // first node uid
        edges_infos.add(second_node); // second node uid
        ++nb_edge;
        ++m_edge_uid_pool;
      }
    }
    faces_infos_index += current_face_nb_node;
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeInfo(Integer& nb_node, Integer nb_face, Int64Array& nodes_infos, Int64ConstArrayView faces_infos)
{
  nodes_infos.reserve(faces_infos.size());
  Integer faces_infos_index = 0;
  std::set<Int64> nodes_set;
  ItemTypeMng* itm = m_item_type_mng;
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {
    Int32 type_id = CheckedConvert::toInt32(faces_infos[faces_infos_index++]);
    Integer current_face_nb_node = itm->typeFromId(type_id)->nbLocalNode(); // face type
    ++faces_infos_index; //face_uid (unused)
    for (auto node_uid : faces_infos.subConstView(faces_infos_index, current_face_nb_node)) {
      nodes_set.insert(node_uid);
    }
    faces_infos_index += current_face_nb_node;
  }
  nb_node = CheckedConvert::toInteger(nodes_set.size());
  for (auto node_uid : nodes_set) {
    nodes_infos.add(node_uid);
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeInfoFromEdge(Integer& nb_node, Integer nb_edge, Int64Array& nodes_infos, Int64ConstArrayView edges_infos)
{
  nodes_infos.reserve(2 * edges_infos.size());
  Integer edges_infos_index = 0;
  std::set<Int64> nodes_set;
  for (Integer i_edge = 0; i_edge < nb_edge; ++i_edge) {
    ++edges_infos_index; //edge_uid (unused)
    nodes_set.insert(edges_infos[edges_infos_index++]);
    nodes_set.insert(edges_infos[edges_infos_index++]);
  }
  nb_node = CheckedConvert::toInteger(nodes_set.size());
  for (auto node_uid : nodes_set) {
    nodes_infos.add(node_uid);
  }
}
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillCellNewInfoNew(Integer nb_cell, Int64ConstArrayView cells_infos, Int64Array& cell_infos2,
                    const std::map<Int64, Int64SharedArray>& cell_to_face_connectivity_info, const std::map<std::pair<Int64, Int64>, Int64>& edge_uid_map)
{
  Integer cell_infos2_size_approx = 1 + 3 * (cells_infos.size() + 2 * nb_cell); // supposes as many faces as nodes as edges...
  cell_infos2.reserve(cell_infos2_size_approx);
  ItemTypeMng* itm = m_item_type_mng;
  Integer cells_infos_index = 0;
  Integer nb_connected_families = hasEdge() ? 3 : 2;
  cell_infos2.add(nb_connected_families); // nb_connected_families (node and face +/- edge)
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    Integer item_type_id = (Integer)cells_infos[cells_infos_index];
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell type
    Int64 current_cell_uid = cells_infos[cells_infos_index];
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell uid
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    //--- Cell node connectivity
    cell_infos2.add(mesh()->nodeFamily()->itemKind()); // connected_family_id: node family
    Integer current_cell_nb_node = it->nbLocalNode();
    cell_infos2.add(current_cell_nb_node); // nb_connected_nodes
    Int64ConstArrayView current_cell_node_uids(current_cell_nb_node, &cells_infos[cells_infos_index]);
    cell_infos2.addRange(current_cell_node_uids); // node ids
    //--- Cell face connectivity
    cell_infos2.add(mesh()->faceFamily()->itemKind()); // connected_family_id: face family
    Integer current_cell_nb_face = it->nbLocalFace();
    cell_infos2.add(current_cell_nb_face); // nb_connected_faces
    Int64ArrayView current_cell_faces = cell_to_face_connectivity_info.find(current_cell_uid)->second.view();
    if (current_cell_nb_face != current_cell_faces.size())
      ARCANE_FATAL("Incoherent number of faces for cell {0}. Expected {1} found {2}",
                   current_cell_uid, current_cell_nb_face, current_cell_faces.size());
    cell_infos2.addRange(current_cell_faces); // face ids
    //-- Cell edge connectivity
    if (hasEdge()) {
      cell_infos2.add(mesh()->edgeFamily()->itemKind()); // connected_family_id: edge family
      Integer current_cell_nb_edge = it->nbLocalEdge();
      cell_infos2.add(current_cell_nb_edge); // nb_connected_edges
      for (int edge_index = 0; edge_index < current_cell_nb_edge; ++edge_index) {
        Int64 first_node = current_cell_node_uids[it->localEdge(edge_index).beginNode()];
        Int64 second_node = current_cell_node_uids[it->localEdge(edge_index).endNode()];
        if (first_node > second_node)
          std::swap(first_node, second_node); // edge may be oriented negatively in the cell
        auto edge_it = edge_uid_map.find(std::make_pair(first_node, second_node));
        if (edge_it == edge_uid_map.end())
          ARCANE_FATAL("Do not find edge with nodes {0}-{1} in edge uid map. Exiting",
                       current_cell_node_uids[it->localEdge(edge_index).beginNode()],
                       current_cell_node_uids[it->localEdge(edge_index).endNode()]);
        cell_infos2.add(edge_it->second);
      }
    }
    cells_infos_index += current_cell_nb_node;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillCellInfo2(Integer nb_cell, Int64ConstArrayView cells_infos, Int64Array& cell_infos2,
               Integer& nb_face, Int64Array& faces_infos, Int64Array& node_uids, bool allow_build_face)
{
  Integer cell_infos2_size_approx = 1 + 2 * (cells_infos.size() + 2 * nb_cell); // supposes as many faces as nodes...
  cell_infos2.reserve(cell_infos2_size_approx);
  // Fill infos
  ItemTypeMng* itm = m_item_type_mng;
  Integer cells_infos_index = 0;
  cell_infos2.add(2); // nb_connected_families (node and face)

  nb_face = 0;
  faces_infos.reserve(2 * cells_infos.size());
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  NodeInFaceSet face_nodes_set;
  std::set<Int64> node_uid_set;
  // Fill face_infos2 from faces_infos
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {

    Integer item_type_id = (Integer)cells_infos[cells_infos_index];
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell type
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell uid
    cell_infos2.add(mesh()->nodeFamily()->itemKind()); // connected_family_id: node family

    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    // Cell node connectivity
    Integer current_cell_nb_node = it->nbLocalNode();
    cell_infos2.add(current_cell_nb_node); // nb_connected_nodes
    Int64ConstArrayView current_cell_node_uids(current_cell_nb_node, &cells_infos[cells_infos_index]);
    cell_infos2.addRange(current_cell_node_uids); // node ids
    if (!allow_build_face)
      node_uid_set.insert(current_cell_node_uids.begin(), current_cell_node_uids.end()); // otherwise node ids will be computed in face info build
    // cell face connectivity
    cell_infos2.add(mesh()->faceFamily()->itemKind()); // connected_family_id: face family
    Integer current_cell_nb_face = it->nbLocalFace();
    cell_infos2.add(current_cell_nb_face); // nb_connected_faces
    for (Integer face_in_cell_index = 0; face_in_cell_index < current_cell_nb_face; ++face_in_cell_index) {
      ItemTypeInfo::LocalFace current_face = it->localFace(face_in_cell_index);
      Integer current_face_nb_node = current_face.nbNode();
      work_face_orig_nodes.resize(current_face_nb_node);
      work_face_sorted_nodes.resize(current_face_nb_node);
      Integer i = 0;
      for (Integer node_in_face_index = 0; node_in_face_index < current_face_nb_node; ++node_in_face_index) {
        work_face_orig_nodes[i++] = current_cell_node_uids[current_face.node(node_in_face_index)];
      }
      mesh_utils::reorderNodesOfFace(work_face_orig_nodes, work_face_sorted_nodes);
      // get face uid: we shouldn't have to guess, it should be given in arguments...
      Int64 current_face_uid = _findFaceUniqueId(work_face_sorted_nodes, face_nodes_set);
      if (current_face_uid == NULL_ITEM_ID) // face not already known add to face info
      {
        current_face_uid = m_face_uid_pool++;
        faces_infos.add(current_face.typeId()); // face info type
        faces_infos.add(current_face_uid); // face info uid
        faces_infos.addRange(work_face_sorted_nodes); // face info node uids
        _addFaceNodes(face_nodes_set, work_face_sorted_nodes, current_face_uid);
        ++nb_face;
      }
      cell_infos2.add(current_face_uid); // face ids
    }
    cells_infos_index += current_cell_nb_node;
  }
  if (!allow_build_face) {
    node_uids.resize(node_uid_set.size());
    Integer index = 0;
    for (Int64 node_uid : node_uid_set)
      node_uids[index++] = node_uid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initializes face relation info.
 *
 * \param source_item_relation_data The destination ItemData for face relations.
 * \param target_item_dependencies_data The source ItemData containing dependencies.
 * \param faces_info The constant array view of face information.
 * @param is_source_relation_data_empty Flag indicating if the source relation data is empty.
 */
void DynamicMeshIncrementalBuilder::
_initFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info)
{
  _fillFaceRelationInfo(source_item_relation_data, target_item_dependencies_data, faces_info, true);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info)
{
  _fillFaceRelationInfo(source_item_relation_data, target_item_dependencies_data, faces_info, false);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data,
                      Int64ConstArrayView faces_info, bool is_source_relation_data_empty)
{
  // face_infos = [{face_type, face_uid, {face_node_uids}]
  Int64UniqueArray face_uids_and_types;
  face_uids_and_types.reserve(2 * source_item_relation_data.nbItems());
  ItemTypeMng* itm = m_item_type_mng;
  for (Integer face_info_index = 0; face_info_index < faces_info.size();) {
    face_uids_and_types.add(faces_info[face_info_index + 1]); // face_uid
    face_uids_and_types.add(faces_info[face_info_index]); // face_type
    Integer type_id = CheckedConvert::toInteger(face_uids_and_types.back());
    face_info_index += (2 + itm->typeFromId(type_id)->nbLocalNode()); // increment and skip info (first & second_node_uid)
  }
  _fillItemRelationInfo(source_item_relation_data, target_item_dependencies_data, face_uids_and_types, is_source_relation_data_empty);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView edges_info)
{
  _fillEdgeRelationInfo(source_item_relation_data, target_item_dependencies_data, edges_info, true);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView edges_info)
{
  _fillEdgeRelationInfo(source_item_relation_data, target_item_dependencies_data, edges_info, false);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data,
                      Int64ConstArrayView edges_info, bool is_source_relation_data_empty)
{
  // edge_infos = [{edge_uid, first_node_uid, second_node_uid}
  ARCANE_ASSERT((source_item_relation_data.nbItems() * 3 == edges_info.size()), ("source_item_relation_data and edges_info size incoherent. Exiting."));
  Int64UniqueArray edge_uids_and_types;
  edge_uids_and_types.reserve(2 * source_item_relation_data.nbItems());
  for (Integer edge_info_index = 0; edge_info_index < edges_info.size();) {
    edge_uids_and_types.add(edges_info[edge_info_index++]); // edge_uid
    edge_uids_and_types.add(IT_Line2); // edge_type
    edge_info_index += 2; // skip info (first & second_node_uid)
  }
  _fillItemRelationInfo(source_item_relation_data, target_item_dependencies_data, edge_uids_and_types, is_source_relation_data_empty);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info)
{
  _fillNodeRelationInfo(source_item_relation_data, target_item_dependencies_data, nodes_info, true);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info)
{
  _fillNodeRelationInfo(source_item_relation_data, target_item_dependencies_data, nodes_info, false);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info, bool is_source_relation_data_empty)
{
  // nodes_info = [node_uids]
  ARCANE_ASSERT((source_item_relation_data.nbItems() == nodes_info.size()), ("source_item_relation_data and nodes_info size incoherent. Exiting."));
  Int64UniqueArray node_uids_and_types(2 * nodes_info.size());
  for (Integer node_index = 0; node_index < nodes_info.size(); ++node_index) {
    node_uids_and_types[2 * node_index] = nodes_info[node_index];
    node_uids_and_types[2 * node_index + 1] = IT_Vertex;
  }
  _fillItemRelationInfo(source_item_relation_data, target_item_dependencies_data, node_uids_and_types, is_source_relation_data_empty);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillItemRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data,
                      Int64ConstArrayView source_item_uids_and_types,
                      bool is_source_item_relation_data_empty)
{
  ARCANE_ASSERT((source_item_relation_data.nbItems() * 2 == source_item_uids_and_types.size()),
                ("source item number incoherent between source_item_relation_data and source_item_uids_and_types. Exiting."));
  // Fill an ItemData containing source to target relations (target to source relations are included in target_item_data)
  auto& source_relation_info = source_item_relation_data.itemInfos();
  const auto& target_dependencies_info = target_item_dependencies_data.itemInfos();
  auto source_family = source_item_relation_data.itemFamily();
  auto target_family = target_item_dependencies_data.itemFamily();
  if (!source_family || !target_family)
    return;
  std::map<Int64, Int64SharedArray> source_to_target_uids;
  Integer nb_families_connected_to_target = CheckedConvert::toInteger(target_dependencies_info[0]);
  Integer target_info_index = 1; // 0 is nb_connected_families

  // Fill map source to target traversing target_item_dependencies_data
  for (; target_info_index < target_dependencies_info.size();) {
    target_info_index++; // current target item_type
    Int64 target_item_uid = target_dependencies_info[target_info_index++]; // current target item uid
    for (Integer family_connected_to_target = 0; family_connected_to_target < nb_families_connected_to_target; ++family_connected_to_target) {
      Int64 family_connected_to_target_kind = target_dependencies_info[target_info_index++]; // current target item connected family kind
      if (family_connected_to_target_kind != source_family->itemKind()) { //this connection info does not concern source family. Skip
        Integer nb_non_read_values = CheckedConvert::toInteger(target_dependencies_info[target_info_index++]); // nb_connected_item on this other family (/= to source family)
        target_info_index += nb_non_read_values;
        continue;
      }
      else {
        Int32 nb_source_item_connected_to_target_item = CheckedConvert::toInt32(target_dependencies_info[target_info_index++]);
        for (Integer source_item_index = 0; source_item_index < nb_source_item_connected_to_target_item; ++source_item_index) {
          source_to_target_uids[target_dependencies_info[source_item_index + target_info_index]].add(target_item_uid);
        }
        target_info_index += nb_source_item_connected_to_target_item;
      }
    }
  }
  // Fill or append source_item_relation_data traversing the map source_to_target_connectivity
  if (is_source_item_relation_data_empty)
    _initEmptyRelationInfo(source_relation_info, source_to_target_uids, source_item_uids_and_types,
                           target_dependencies_info.size(), target_family);
  else
    _appendInitializedRelationInfo(source_relation_info, source_to_target_uids, source_item_uids_and_types,
                                   target_dependencies_info.size(), target_family);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initEmptyRelationInfo(Int64Array& source_relation_info, std::map<Int64, Int64SharedArray>& source_to_target_uids, Int64ConstArrayView source_item_uids_and_types,
                       Integer approx_relation_size,
                       IItemFamily const* target_family)
{
  if (!source_relation_info.empty())
    source_relation_info.clear();
  source_relation_info.reserve(approx_relation_size); // clearly too much elements...but only available info. Do better ?
  source_relation_info.add(1); // nb_connected_families
  // Fill source_item_relation_data
  for (Integer source_item_uids_and_types_index = 0; source_item_uids_and_types_index < source_item_uids_and_types.size(); source_item_uids_and_types_index += 2) {
    Int64 source_item_uid = source_item_uids_and_types[source_item_uids_and_types_index];
    source_relation_info.add(source_item_uids_and_types[source_item_uids_and_types_index + 1]); // item type
    source_relation_info.add(source_item_uid); // item uid
    source_relation_info.add(target_family->itemKind()); // connected family kind
    auto& connected_items = source_to_target_uids[source_item_uid];
    source_relation_info.add(connected_items.size()); // nb_connected_item
    source_relation_info.addRange(connected_items.view()); // connected_items uids
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendInitializedRelationInfo(Int64Array& source_relation_info, std::map<Int64, Int64SharedArray>& source_to_target_uids, Int64ConstArrayView source_item_uids_and_types,
                               Integer approx_relation_size,
                               IItemFamily const* target_family)
{
  // Create a working array to build the consolidated info. This array will be copied in to source_relation_info at the end.
  Int64UniqueArray source_relation_info_wrk_copy;
  source_relation_info_wrk_copy.reserve(source_relation_info.size() + approx_relation_size);
  std::set<Int64> treated_items; // To detect eventual source_items not already present in source_relation_info
  Integer source_relation_info_index = 0;
  Integer nb_connected_family = CheckedConvert::toInteger(source_relation_info[source_relation_info_index++]);
  source_relation_info_wrk_copy.add(nb_connected_family + 1); // adding a new connected family
  for (; source_relation_info_index < source_relation_info.size();) {
    source_relation_info_wrk_copy.add(source_relation_info[source_relation_info_index++]); // item type
    Int64 source_item_uid = source_relation_info[source_relation_info_index++]; // get item uid
    source_relation_info_wrk_copy.add(source_item_uid);
    treated_items.insert(source_item_uid);
    for (Integer connected_family_index = 0; connected_family_index < nb_connected_family; ++connected_family_index) {
      source_relation_info_wrk_copy.add(source_relation_info[source_relation_info_index++]); // family kind
      Integer nb_connected_elements = CheckedConvert::toInteger(source_relation_info[source_relation_info_index++]); // nb connected elements
      source_relation_info_wrk_copy.add(nb_connected_elements);
      source_relation_info_wrk_copy.addRange(source_relation_info.subConstView(source_relation_info_index, nb_connected_elements));
      source_relation_info_index += nb_connected_elements;
    }
    // Add new connection info
    source_relation_info_wrk_copy.add(target_family->itemKind()); // family kind
    const auto& connected_items = source_to_target_uids[source_item_uid]; // get connected items
    source_relation_info_wrk_copy.add(connected_items.size()); // nb connected elements
    source_relation_info_wrk_copy.addRange(connected_items);
  }
  // Find eventual source_items not already present in source_relation_info : todo check this can happen
  for (Integer source_item_uids_and_types_index = 0; source_item_uids_and_types_index < source_item_uids_and_types.size(); source_item_uids_and_types_index += 2) {
    Int64 source_item_uid = source_item_uids_and_types[source_item_uids_and_types_index];
    if (treated_items.find(source_item_uid) == treated_items.end()) {
      source_relation_info_wrk_copy.add(source_item_uids_and_types[source_item_uids_and_types_index + 1]); // item type
      source_relation_info_wrk_copy.add(source_item_uid); // item uid
      source_relation_info_wrk_copy.add(target_family->itemKind()); // family kind
      const auto& connected_items = source_to_target_uids[source_item_uid]; // get connected items
      source_relation_info_wrk_copy.add(connected_items.size()); // nb connected elements
      source_relation_info_wrk_copy.addRange(connected_items);
      // Padd this item info with family info : we added only one family, add void values for the other nb_connected_family families
      for (Integer connected_family_index = 0; connected_family_index < nb_connected_family; ++connected_family_index) {
        source_relation_info_wrk_copy.add(0); // family id (does not matter)
        source_relation_info_wrk_copy.add(0); // nb_connected_elements for this family
      }
    }
  }
  // Copy working array into destination array
  source_relation_info.resize(source_relation_info_wrk_copy.size());
  std::copy(source_relation_info_wrk_copy.begin(), source_relation_info_wrk_copy.end(), source_relation_info.begin());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DynamicMeshIncrementalBuilder::
_findFaceUniqueId(Int64ConstArrayView work_face_sorted_nodes, NodeInFaceSet& face_nodes_set)
{
  // check if first node exist
  Integer index = 0;
  //  return _findFaceInFaceNodesSet(face_nodes_set,index,work_face_sorted_nodes,std::make_shared<NodeInFace>(NULL_ITEM_ID));
  return _findFaceInFaceNodesSet(face_nodes_set, index, work_face_sorted_nodes, NodeInFacePtr(NULL_ITEM_ID));
}

/*---------------------------------------------------------------------------*/

Int64 DynamicMeshIncrementalBuilder::
_findFaceInFaceNodesSet(const NodeInFaceSet& face_nodes_set, Integer index, Int64ConstArrayView face_nodes, NodeInFacePtr node)
{
  if (index < face_nodes.size()) {
    auto found_node = std::find(face_nodes_set.begin(), face_nodes_set.end(), NodeInFacePtr(face_nodes[index]));
    if (found_node == face_nodes_set.end())
      return node->faceUid();
    else
      return _findFaceInFaceNodesSet((*found_node)->nextNodeSet(), ++index, face_nodes, *found_node);
  }
  else
    return node->faceUid();
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_addFaceNodes(NodeInFaceSet& face_nodes_set, Int64ConstArrayView face_nodes, Int64 face_uid)
{
  Integer index = 0;
  //  _addFaceInFaceNodesSet(face_nodes_set, index,face_nodes, std::make_shared<NodeInFace>(NULL_ITEM_ID), face_uid);
  _addFaceInFaceNodesSet(face_nodes_set, index, face_nodes, NodeInFacePtr(NULL_ITEM_ID), face_uid);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_addFaceInFaceNodesSet(NodeInFaceSet& face_nodes_set, Integer index, Int64ConstArrayView face_nodes, NodeInFacePtr node, Int64 face_uid)
{
  if (index < face_nodes.size()) {
    //      auto next_node = std::make_shared<NodeInFace>(face_nodes[index]);
    auto next_node = _insertNode(face_nodes_set, face_nodes[index]);
    //      debug() << "Add node " << face_nodes[index] << " in set " << &face_nodes_set;
    //      debug() << "set size " << face_nodes_set.size();
    _addFaceInFaceNodesSet(next_node->nextNodeSet(), ++index, face_nodes, next_node, face_uid);
    //      else
    //        _addFaceInFaceNodesSet((*insertion_return.first)->nextNodeSet(),++index,face_nodes,*insertion_return.first,face_uid); // pbe du const. Change from set to map ?
  }
  else {
    node->setFaceUid(face_uid);
    //      debug() << "Set face uid " << face_uid << " for node " << node->m_uid;
  }
}

DynamicMeshIncrementalBuilder::NodeInFacePtr&
DynamicMeshIncrementalBuilder::_insertNode(NodeInFaceSet& face_nodes_set, Int64 inserted_node_uid)
{
  NodeInFacePtr new_node(inserted_node_uid);
  auto found_node = std::find(face_nodes_set.begin(), face_nodes_set.end(), new_node);
  if (found_node == face_nodes_set.end()) {
    face_nodes_set.push_back(std::move(new_node));
    return face_nodes_set.back();
  }
  else
    return *found_node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des items du maillage parent au en tant que maille au maillage actuel.
 *
 * \param items items à ajouter (vu depuis la maillage parent)
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 */
void DynamicMeshIncrementalBuilder::
addParentCells(const ItemVectorView& items)
{
  const Integer nb_cell = items.size();

  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    m_one_mesh_item_adder->addOneParentItem(items[i_cell], IK_Cell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Ajoute des mailles au maillage actuel.
 *
 * \param mesh_nb_cell nombre de mailles à ajouter
 * \param cells_infos infos sur les maillage (voir IMesh::allocateMesh())
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 * \param cells en retour, si non vide, contient les mailles créées.
 */
void DynamicMeshIncrementalBuilder::
addHChildrenCells(Cell hParent_cell, Integer nb_cell, Int64ConstArrayView cells_infos,
                  Integer sub_domain_id, Int32ArrayView cells,
                  bool allow_build_face)
{
  ItemTypeMng* itm = m_item_type_mng;
  CellFamily& cell_family = m_mesh->trueCellFamily();

  debug() << "[DynamicMeshIncrementalBuilder::addHChildrenCells] ADD CELLS mesh=" << m_mesh->name() << " nb_cell=" << nb_cell;
  Integer cells_infos_index = 0;
  bool add_to_cells = cells.size() != 0;
  if (add_to_cells && nb_cell != cells.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'cells' has to have same size as number of cells");
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    debug(Trace::Highest) << "\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]cell " << i_cell << "/" << nb_cell;

    ItemTypeId item_type_id{ (Int16)cells_infos[cells_infos_index] };
    debug(Trace::Highest) << "\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]cells_infos[" << cells_infos_index << "]=" << cells_infos[cells_infos_index] << ", type_id=" << item_type_id;
    ++cells_infos_index;

    Int64 cell_unique_id = cells_infos[cells_infos_index];
    debug(Trace::Highest) << "\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]cells_infos[" << cells_infos_index << "]=" << cells_infos[cells_infos_index] << ", uid=" << cell_unique_id;
    ++cells_infos_index;

    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_cell_nb_node = it->nbLocalNode();
    debug(Trace::Highest) << "\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]DM ADD_CELL uid=" << cell_unique_id << " type_id=" << item_type_id
                          << " nb_node=" << current_cell_nb_node;
    Int64ConstArrayView current_cell_nodes_uid(current_cell_nb_node, &cells_infos[cells_infos_index]);
    for (Integer i = 0; i < current_cell_nb_node; ++i)
      debug(Trace::Highest) << "\t\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]DM NODE uid=" << current_cell_nodes_uid[i];

    ItemInternal* cell = m_one_mesh_item_adder->addOneCell(item_type_id,
                                                           cell_unique_id,
                                                           sub_domain_id,
                                                           current_cell_nodes_uid,
                                                           allow_build_face);

    cell_family._addParentCellToCell(cell, hParent_cell);
    if (add_to_cells)
      cells[i_cell] = cell->localId();
    cells_infos_index += current_cell_nb_node;
  }
  cell_family._addChildrenCellsToCell(hParent_cell, cells);

  debug() << "[DynamicMeshIncrementalBuilder::addHChildrenCells] done";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des items du maillage parent au en tant qu'items fantomes au maillage actuel.
 *
 * \param items items à ajouter (vu depuis la maillage parent)
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 */
void DynamicMeshIncrementalBuilder::
addParentItems(const ItemVectorView& items, const eItemKind submesh_kind)
{
  const Integer nb_item = items.size();

  for (Integer i_item = 0; i_item < nb_item; ++i_item) {
    m_one_mesh_item_adder->addOneParentItem(items[i_item], submesh_kind, false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des noeuds au maillage actuel.
 *
 * \param nodes_uid identifiant unique des noeuds à créer.
 * \param owner sous-domaine auquel les noeuds appartiendront.
 * \param nodes en retour, si non vide, contient les noeuds correspodants aux \a nodes_uid
 */
void DynamicMeshIncrementalBuilder::
addNodes(Int64ConstArrayView nodes_uid, Integer owner, Int32ArrayView nodes)
{
  Integer nb_node_to_add = nodes_uid.size();
  bool add_to_nodes = nodes.size() != 0;
  if (add_to_nodes && nb_node_to_add != nodes.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'nodes' has to have same size as number of nodes");
  for (Integer i_node = 0; i_node < nb_node_to_add; ++i_node) {
    ItemInternal* node = m_one_mesh_item_adder->addOneNode(nodes_uid[i_node], owner);
    if (add_to_nodes)
      nodes[i_node] = node->localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des noeuds au maillage actuel. Utilise l'ajout d'item générique basé sur dépendances entre familles.
 *
 * \param nodes_uid identifiant unique des noeuds à créer.
 * \param owner sous-domaine auquel les noeuds appartiendront.
 * \param nodes en retour, si non vide, contient les noeuds correspodants aux \a nodes_uid
 */
void DynamicMeshIncrementalBuilder::
addNodes2(Int64ConstArrayView nodes_uid, Integer owner, Int32ArrayView nodes)
{
  Integer nb_node_to_add = nodes_uid.size();
  bool add_to_nodes = nodes.size() != 0;
  if (add_to_nodes && nb_node_to_add != nodes.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'nodes' has to have same size as number of nodes");
  ItemDataList item_info_list;
  ItemData& node_data = item_info_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
                                                nb_node_to_add, 2 * nb_node_to_add + 1, nodes, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), owner);
  Int64Array& node_infos2 = node_data.itemInfos();
  // Copy node_info : to decrease the footprint of the copy, use references instead of int64
  _fillNodeNewInfo(nodes_uid, node_infos2);

  addItems(item_info_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeNewInfo(Int64ConstArrayView node_uids, Int64ArrayView node_infos2)
{
  node_infos2[0] = 0; // nb_connected_families
  Integer index = 0;
  for (auto node_uid : node_uids) {
    node_infos2[2 * index + 1] = IT_Vertex; // type
    node_infos2[2 * index + 2] = node_uid; // id
    index++;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajout générique d'items d'un ensemble de famille pour lesquelles on fournit un ItemData.
 *
 * L'objet \a ItemDataList est une map <family_index,ItemData> où family_index est pris égal à l'item_kind de la famille
 * et où ItemData aggrège les informations id/connectivités des items Le tableau item_infos (ItemData::itemInfos()) à la structure suivante :
 * item_infos[0]   = nb_connected_families // Only constitutive (owning) connections.
 * item_infos[i]   = first_item_type
 * item_infos[i+1] = first_item_uid
 * item_infos[i+2] = first_family_id
 * item_infos[i+3] = nb_connected_items_in_first_family
 * item_infos[i+4...i+n] = first_family items uids
 * item_infos[i+n+1] = second_family_id
 * item_infos[i+n+1...i+m] = second_family items uids
 * item_infos[i+m+1] = second_item_uid
 * ...idem first item
 * La méthode parcours le graphe des connectivités du maillage pour créer les items de toute les familles
 * La méthode addFamilyItems(ItemInfo&) crée les items d'une famille donnée
 */

void DynamicMeshIncrementalBuilder::
addItems(ItemDataList& item_info_list)
{
  // Add items : item construction done from leaves to root
  _addItemsOrRelations(item_info_list, IItemFamilyNetwork::InverseTopologicalOrder);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addItems(ItemDataList& item_info_list, ItemDataList& item_relation_info_list)
{
  if (item_info_list.size() == 0 || item_relation_info_list.size() == 0)
    return;
  addItems(item_info_list);
  addRelations(item_relation_info_list);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addRelations(ItemDataList& item_relation_list)
{
  // Add relations : items are already added. Do it from root to leaves
  _addItemsOrRelations(item_relation_list, IItemFamilyNetwork::TopologicalOrder);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_addItemsOrRelations(ItemDataList& info_list, IItemFamilyNetwork::eSchedulingOrder family_graph_traversal_order)
{
  // Add relations : items are already added. Do it from root to leaves
  if (info_list.size() == 0)
    return;
  m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family) {
    ItemData i_infos = info_list[family->itemKind()];
    this->addFamilyItems(i_infos);
  },
                                        family_graph_traversal_order); // item construction done from leaves to root
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajout générique d'items d'une famille, décrite par son ItemInfo
 *
 */

void DynamicMeshIncrementalBuilder::
addFamilyItems(ItemData& item_data)
{
  if (item_data.nbItems() == 0)
    return;
  bool add_to_items = item_data.itemLids().size() != 0;
  if (add_to_items && item_data.nbItems() != item_data.itemLids().size())
    throw ArgumentException(A_FUNCINFO,
                            "return array containing item lids has to have be of size number of items");
  // Prepare info and call OneMeshItemAdder->addOneItem
  Int64 item_uid;
  Int64ConstArrayView item_infos = item_data.itemInfos();
  Integer nb_connected_family = CheckedConvert::toInteger(item_infos[0]);
  if (nb_connected_family == 0)
    return;
  Int64UniqueArray connectivity_info;
  Integer nb_item_info = 0;
  Integer item_index = 0;
  for (Integer info_index = 1; info_index < item_data.itemInfos().size();) {
    ItemTypeId item_type_id{ (Int16)item_infos[info_index++] }; // item_type
    item_uid = item_infos[info_index++]; // item_uid
    Integer item_info_begin_index = info_index;
    for (Integer connected_family_index = 0; connected_family_index < nb_connected_family; ++connected_family_index) {
      Integer current_index = CheckedConvert::toInteger(item_infos[info_index + 1]);
      Integer nb_item_info_increment = 2 + current_index;
      nb_item_info += nb_item_info_increment; // family_id , nb_connected_elements
      info_index += nb_item_info_increment; // pointing on next family id
    }
    ItemInternal* item = m_one_mesh_item_adder->addOneItem2(item_data.itemFamily(),
                                                            item_data.itemFamilyModifier(),
                                                            item_type_id,
                                                            item_uid,
                                                            item_data.itemOwners()[item_index],
                                                            item_data.subDomainId(),
                                                            nb_connected_family,
                                                            item_infos.subView(item_info_begin_index, nb_item_info));
    if (add_to_items)
      item_data.itemLids()[item_index++] = item->localId();
    nb_item_info = 0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au maillage actuel.
 *
 * \param nb_face nombre de faces à ajouter
 * \param faces_infos infos sur les faces (voir IMesh::allocateCells())
 * \param sub_domain_id sous-domaine auquel les faces appartiendront
 * \param faces en retour, si non vide, contient les faces créées.
 */
void DynamicMeshIncrementalBuilder::
addFaces(Integer nb_face, Int64ConstArrayView faces_infos,
         Int32 rank, Int32ArrayView faces)
{
  ItemTypeMng* itm = m_item_type_mng;

  bool add_to_faces = faces.size() != 0;
  if (add_to_faces && nb_face != faces.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'faces' has to have same size as number of faces");

  Integer faces_infos_index = 0;
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {

    ItemTypeId item_type_id = ItemTypeId::fromInteger(faces_infos[faces_infos_index]);
    ++faces_infos_index;
    Int64 face_unique_id = faces_infos[faces_infos_index];
    ++faces_infos_index;

    ItemTypeInfo* it = itm->typeFromId(item_type_id);

    Integer current_face_nb_node = it->nbLocalNode();
    Int64ConstArrayView nodes_uid(current_face_nb_node, &faces_infos[faces_infos_index]);
    faces_infos_index += current_face_nb_node;

    ItemInternal* face = m_one_mesh_item_adder->addOneFace(item_type_id, face_unique_id, rank, nodes_uid);

    if (add_to_faces)
      faces[i_face] = face->localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au maillage actuel. Utilise l'ajout d'item générique basé sur dépendances entre familles.
 *
 * \param nb_face nombre de faces à ajouter
 * \param faces_infos infos sur les faces (voir IMesh::allocateCells())
 * \param sub_domain_id sous-domaine auquel les faces appartiendront
 * \param faces en retour, si non vide, contient les faces créées.
 */
void DynamicMeshIncrementalBuilder::
addFaces2(Integer nb_face, Int64ConstArrayView faces_infos,
          Integer sub_domain_id, Int32ArrayView faces)
{
  bool add_to_faces = faces.size() != 0;
  if (add_to_faces && nb_face != faces.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'faces' has to have same size as number of faces");
  // Prepare face infos 2
  Integer face_infos2_size = faces_infos.size() + 1 + 2 * nb_face; // additive info nb_connected_family + nb_face * (family_id + nb_connected_element)
  ItemDataList item_info_list;
  ItemData& face_data = item_info_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),
                                                nb_face, face_infos2_size, faces, &mesh()->trueFaceFamily(), &mesh()->trueFaceFamily(), sub_domain_id);

  Int64ArrayView face_infos2 = face_data.itemInfos().view();
  Int64UniqueArray node_uids;
  Int64UniqueArray edge_uids;

  _fillFaceInfo2(nb_face, faces_infos, face_infos2, node_uids);

  // Prepare edge data if needed
  //  if (m_has_edge) {
  //    Integer nb_edge_to_add = edge_uids.size();
  //    Int32UniqueArray edge_lids(nb_edge_to_add);
  //    ItemData& edge_data = item_info_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),
  //          nb_edge_to_add,2*nb_edge_to_add+1,edge_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  //    Int64Array& edge_infos2 = edge_data.itemInfos();
  //    _fillEdgeInfo2(nb_edge_to_add,edge_uids,edge_infos2);
  //  }

  // Prepare node data (mutualize)
  Integer nb_node_to_add = node_uids.size();
  Int32UniqueArray node_lids(nb_node_to_add);
  ItemData& node_data = item_info_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
                                                nb_node_to_add, 2 * nb_node_to_add + 1, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  Int64Array& node_infos2 = node_data.itemInfos();
  _fillNodeNewInfo(node_uids, node_infos2);

  // Launch node and face creation
  addItems(item_info_list);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addFaces3(Integer nb_face, Int64ConstArrayView faces_infos,
          Integer sub_domain_id, Int32ArrayView faces)
{
  bool add_to_faces = faces.size() != 0;
  if (add_to_faces && nb_face != faces.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'faces' has to have same size as number of faces");
  ItemDataList item_data_list;
  ItemDataList item_relation_data_list;
  // item info_size cannot be guessed
  Int64UniqueArray edges_infos;
  Int64UniqueArray nodes_infos;
  Int64UniqueArray node_uids;
  Int32UniqueArray edge_lids;
  Int32UniqueArray node_lids;
  Integer nb_edge = 0;
  Integer nb_node = 0;
  std::map<std::pair<Int64, Int64>, Int64> edge_uid_map;

  // Fill classical info
  _fillEdgeInfo(nb_edge, nb_face, edges_infos, faces_infos, edge_uid_map);
  _fillNodeInfo(nb_node, nb_face, nodes_infos, faces_infos);

  // Face info New
  ItemData& face_data = item_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()), nb_face, 0, faces, &mesh()->trueFaceFamily(), &mesh()->trueFaceFamily(), sub_domain_id);
  _fillFaceNewInfoNew(nb_face, faces_infos, face_data.itemInfos(), edge_uid_map);

  // Edge info New
  if (hasEdge()) {
    Integer edge_info_size = 1 + nb_edge * 6; //New info = Nb_connected_family + for each edge : type, uid, connected_family_kind, nb_node_connected, first_node_uid, second_node_uid
    ItemData& edge_data = item_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()), nb_edge, edge_info_size, edge_lids, &mesh()->trueEdgeFamily(), &mesh()->trueEdgeFamily(), sub_domain_id);
    edge_lids.resize(item_data_list[m_mesh->edgeFamily()->itemKind()].nbItems());
    _fillEdgeNewInfoNew(nb_edge, edges_infos, edge_data.itemInfos()); // reprendre fillEdgeInfo et voir l'appel depuis add Edges + attention ordre des noeuds,
    ItemData& edge_relation_data = item_relation_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()), nb_edge, edge_info_size, edge_lids, &mesh()->trueEdgeFamily(), &mesh()->trueEdgeFamily(), sub_domain_id);
    _initEdgeRelationInfo(edge_relation_data, face_data, edges_infos); // edge-face relation
  }

  // Node info New
  Integer node_info_size = 1 + nb_node * 2; //New info = Nb_connected_family + for each node : type, uid (no connected family)
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()), nb_node, node_info_size, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  node_lids.resize(item_data_list[m_mesh->nodeFamily()->itemKind()].nbItems());
  _fillNodeNewInfo(nodes_infos, node_data.itemInfos());
  ItemData& node_relation_data = item_relation_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()), nb_node, node_info_size, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  _initNodeRelationInfo(node_relation_data, face_data, nodes_infos); // node-face relation

  // Launch node, face and edges (not compulsory) creation
  addItems(item_data_list, item_relation_data_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceNewInfoNew(Integer nb_face, Int64ConstArrayView faces_infos, Int64Array& face_infos2, const std::map<std::pair<Int64, Int64>, Int64>& edge_uid_map)
{
  ItemTypeMng* itm = m_item_type_mng;
  Integer nb_connected_family = hasEdge() ? 2 : 1;
  Integer face_infos2_size_approx = 1 + 2 * (faces_infos.size() + 2 * nb_face); // supposes as many faces as edges...
  face_infos2.reserve(face_infos2_size_approx);
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  Integer faces_infos_index = 0;
  //  Integer face_infos2_index = 0;
  std::set<Int64> node_uids_set;
  face_infos2.add(nb_connected_family); // nb_connected_families nodes +/- Edges !!
  // Fill face_infos2 from faces_infos
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {
    Integer current_face_type = (Integer)faces_infos[faces_infos_index++];
    Integer current_face_uid = (Integer)faces_infos[faces_infos_index++];
    face_infos2.add(current_face_type); // type
    face_infos2.add(current_face_uid); // uid
    //--Node connectivity
    face_infos2.add(mesh()->nodeFamily()->itemKind()); // connected_family_id
    ItemTypeInfo* it = itm->typeFromId(current_face_type);
    Integer current_face_nb_node = it->nbLocalNode();
    face_infos2.add(current_face_nb_node); // nb_connected_nodes
    // Nodes are not necessarily sorted (if caller is addFaces...)
    Int64ConstArrayView current_face_node_uids(current_face_nb_node, &faces_infos[faces_infos_index]);
    work_face_orig_nodes.resize(current_face_nb_node);
    work_face_sorted_nodes.resize(current_face_nb_node);
    work_face_orig_nodes.copy(current_face_node_uids);
    mesh_utils::reorderNodesOfFace(work_face_orig_nodes, work_face_sorted_nodes);
    face_infos2.addRange(work_face_sorted_nodes); // connected node uids
    faces_infos_index += current_face_nb_node;
    //--Edge Connectivity
    if (hasEdge()) {
      face_infos2.add(mesh()->edgeFamily()->itemKind());
      face_infos2.add(it->nbLocalEdge()); // nb_connected_edges
      for (int edge_index = 0; edge_index < it->nbLocalEdge(); ++edge_index) {
        Int64 first_node = work_face_sorted_nodes[it->localEdge(edge_index).beginNode()];
        Int64 second_node = work_face_sorted_nodes[it->localEdge(edge_index).endNode()];
        if (first_node > second_node)
          std::swap(first_node, second_node); // edge may be oriented negatively in the face
        auto edge_it = edge_uid_map.find(std::make_pair(first_node, second_node));
        if (edge_it == edge_uid_map.end())
          ARCANE_FATAL("Do not find edge with nodes {0}-{1} in edge uid map. Exiting",
                       work_face_sorted_nodes[it->localEdge(edge_index).beginNode()],
                       work_face_sorted_nodes[it->localEdge(edge_index).endNode()]);
        face_infos2.add(edge_it->second); // connected edge uid
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceInfo2(Integer nb_face, Int64ConstArrayView faces_infos, Int64ArrayView face_infos2, Int64Array& node_uids)
{
  ItemTypeMng* itm = m_item_type_mng;
  Integer faces_infos_index = 0;
  Integer face_infos2_index = 0;
  face_infos2[face_infos2_index++] = 1; // nb_connected_families (node only)
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  std::set<Int64> node_uids_set;
  // Fill face_infos2 from faces_infos
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {

    Integer item_type_id = (Integer)faces_infos[faces_infos_index];
    face_infos2[face_infos2_index++] = faces_infos[faces_infos_index++]; // type
    face_infos2[face_infos2_index++] = faces_infos[faces_infos_index++]; // uid
    face_infos2[face_infos2_index++] = mesh()->nodeFamily()->itemKind(); // connected_family_id

    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_face_nb_node = it->nbLocalNode();
    face_infos2[face_infos2_index++] = current_face_nb_node; // nb_connected_items
    Int64ConstArrayView current_face_node_uids(current_face_nb_node, &faces_infos[faces_infos_index]);
    work_face_orig_nodes.resize(current_face_nb_node);
    work_face_sorted_nodes.resize(current_face_nb_node);
    work_face_orig_nodes.copy(current_face_node_uids);
    mesh_utils::reorderNodesOfFace(work_face_orig_nodes, work_face_sorted_nodes);
    face_infos2.subView(face_infos2_index, current_face_nb_node).copy(work_face_sorted_nodes); // connected node uids
    faces_infos_index += current_face_nb_node;
    face_infos2_index += current_face_nb_node;
    node_uids_set.insert(current_face_node_uids.begin(), current_face_node_uids.end());
  }
  node_uids.resize(node_uids_set.size());
  Integer index = 0;
  for (Int64 node_uid : node_uids_set)
    node_uids[index++] = node_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds edges to the current mesh.
 *
 * \param nb_face number of faces to add
 * \param edge_infos edge information (see IMesh::allocateCells() without type identifier)
 * \param sub_domain_id sub-domain to which the edges will belong
 * \param edges returns, if not empty, contains the created edges.
 */
void DynamicMeshIncrementalBuilder::
addEdges(Integer nb_edge, Int64ConstArrayView edge_infos,
         Integer sub_domain_id, Int32ArrayView edges)
{
  bool add_to_edges = edges.size() != 0;
  if (add_to_edges && nb_edge != edges.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'edges' has to have same size as number of edges");

  Integer edge_info_index = 0;
  for (Integer i_edge = 0; i_edge < nb_edge; ++i_edge) {

    Int64 edge_unique_id = edge_infos[edge_info_index];
    ++edge_info_index;

    Int64ConstArrayView nodes_uid(2, &edge_infos[edge_info_index]);
    edge_info_index += 2;

    ItemInternal* edge = m_one_mesh_item_adder->addOneEdge(edge_unique_id, sub_domain_id, nodes_uid);

    if (add_to_edges)
      edges[i_edge] = edge->localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds edges to the current mesh. Uses generic item addition based on family dependencies.
 *
 * \param nb_face number of faces to add
 * \param edge_infos edge information (see IMesh::allocateCells() without type identifier)
 * \param sub_domain_id sub-domain to which the edges will belong
 * \param edges returns, if not empty, contains the created edges.
 */
void DynamicMeshIncrementalBuilder::
addEdges2(Integer nb_edge, Int64ConstArrayView edge_infos,
          Integer sub_domain_id, Int32ArrayView edges)
{
  bool add_to_edges = edges.size() != 0;
  if (add_to_edges && nb_edge != edges.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'edges' has to have same size as number of edges");

  // Prepare face infos 2
  Integer edge_infos2_size = edge_infos.size() + 1 + 2 * nb_edge; // additive info nb_connected_family + nb_face * (family_id + nb_connected_element)
  ItemDataList item_info_list;
  ItemData& edge_data = item_info_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),
                                                nb_edge, edge_infos2_size, edges, &mesh()->trueEdgeFamily(), &mesh()->trueEdgeFamily(), sub_domain_id);

  Int64ArrayView edge_infos2 = edge_data.itemInfos().view();
  Int64UniqueArray node_uids;

  _fillEdgeInfo2(nb_edge, edge_infos, edge_infos2, node_uids);

  // Prepare node data (mutualize)
  Integer nb_node_to_add = node_uids.size();
  Int32UniqueArray node_lids(nb_node_to_add);
  ItemData& node_data = item_info_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
                                                nb_node_to_add, 2 * nb_node_to_add + 1, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  Int64Array& node_infos2 = node_data.itemInfos();
  _fillNodeNewInfo(node_uids, node_infos2);

  // Launch node and face creation
  addItems(item_info_list);
}

void DynamicMeshIncrementalBuilder::
addEdges3(Integer nb_edge, Int64ConstArrayView edges_infos,
          Integer sub_domain_id, Int32ArrayView edges)
{
  bool add_to_edges = edges.size() != 0;
  if (add_to_edges && nb_edge != edges.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'edges' has to have same size as number of edges");
  ItemDataList item_data_list;
  ItemDataList item_relation_data_list;
  // item info_size cannot be guessed
  Int64UniqueArray nodes_infos;
  Int32UniqueArray node_lids;
  Integer nb_node = 0;

  // Fill classical info
  _fillNodeInfoFromEdge(nb_node, nb_edge, nodes_infos, edges_infos);

  // Edge new info : edge data stores edge uids and dependency connectivities
  ItemData& edge_data = item_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()), nb_edge, 0, edges, &mesh()->trueEdgeFamily(), &mesh()->trueEdgeFamily(), sub_domain_id);
  _fillEdgeNewInfoNew(nb_edge, edges_infos, edge_data.itemInfos());

  // Node new info & relationsm
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()), nb_node, 0, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  node_lids.resize(nb_node);
  _fillNodeNewInfo(nodes_infos, node_data.itemInfos());
  ItemData& node_relations_data = item_relation_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()), nb_node, 0, node_lids, &mesh()->trueNodeFamily(), &mesh()->trueNodeFamily(), sub_domain_id);
  _initNodeRelationInfo(node_relations_data, edge_data, nodes_infos); // node-edge relation

  // Launch node and edges creation
  addItems(item_data_list, item_relation_data_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillEdgeInfo2(Integer nb_edge, Int64ConstArrayView edges_infos, Int64ArrayView edge_infos2, Int64Array& node_uids)
{
  Integer edges_infos_index = 0;
  Integer edge_infos2_index = 0;
  std::set<Int64> node_uids_set;

  // Fill edge_infos2 from edges_infos
  edge_infos2[edge_infos2_index++] = 1; // nb_connected_families (node only)

  for (Integer i_edge = 0; i_edge < nb_edge; ++i_edge) {
    edge_infos2[edge_infos2_index++] = IT_Line2; // type
    edge_infos2[edge_infos2_index++] = edges_infos[edges_infos_index++]; // uid
    edge_infos2[edge_infos2_index++] = mesh()->nodeFamily()->itemKind(); // connected_family_id
    edge_infos2[edge_infos2_index++] = 2; // nb_connected_items Attention : ordre des noeuds !! TODO
    edge_infos2[edge_infos2_index++] = edges_infos[edges_infos_index]; // first connected node uid
    node_uids_set.insert(edges_infos[edges_infos_index++]);
    edge_infos2[edge_infos2_index++] = edges_infos[edges_infos_index]; // second connected node uid
    node_uids_set.insert(edges_infos[edges_infos_index++]);
  }
  node_uids.resize(node_uids_set.size());
  Integer index = 0;
  for (Int64 node_uid : node_uids_set)
    node_uids[index++] = node_uid;
}

void DynamicMeshIncrementalBuilder::
_fillEdgeNewInfoNew(Integer nb_edge, Int64ConstArrayView edges_infos, Int64ArrayView edges_new_infos)
{
  Integer edges_infos_index = 0;
  Integer edges_new_infos_index = 0;

  // Fill edge_infos2 from edges_infos
  edges_new_infos[edges_new_infos_index++] = 1; // nb_connected_families (node only)

  for (Integer i_edge = 0; i_edge < nb_edge; ++i_edge) {
    edges_new_infos[edges_new_infos_index++] = IT_Line2; // type
    edges_new_infos[edges_new_infos_index++] = edges_infos[edges_infos_index++]; // uid
    edges_new_infos[edges_new_infos_index++] = mesh()->nodeFamily()->itemKind(); // connected_family_id
    edges_new_infos[edges_new_infos_index++] = 2; // nb_connected_items
    Int64 first_node = edges_infos[edges_infos_index++];
    Int64 second_node = edges_infos[edges_infos_index++];
    if (first_node > second_node)
      std::swap(first_node, second_node);
    edges_new_infos[edges_new_infos_index++] = first_node; // first connected node uid
    edges_new_infos[edges_new_infos_index++] = second_node; // second connected node uid
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
readFromDump()
{
  // Recalculate the max uniqueId() of faces to know what value
  // to use in case of face creation.
  ItemInternalMap& faces_map = m_mesh->facesMap();
  Int64 max_uid = -1;
  faces_map.eachItem([&](Item face) {
    if (face.uniqueId() > max_uid)
      max_uid = face.uniqueId();
  });
  m_one_mesh_item_adder->setNextFaceUid(max_uid + 1);

  if (m_has_edge) {
    ItemInternalMap& edges_map = m_mesh->edgesMap();
    Int64 max_uid = -1;
    edges_map.eachItem([&](Item edge) {
      if (edge.uniqueId() > max_uid)
        max_uid = edge.uniqueId();
    });
    m_one_mesh_item_adder->setNextEdgeUid(max_uid + 1);
  }

  info(5) << "NEXT FACE UID mesh=" << m_mesh->name() << " Next=" << m_one_mesh_item_adder->nextFaceUid();
  info(5) << "NEXT EDGE UID mesh=" << m_mesh->name() << " Next=" << m_one_mesh_item_adder->nextEdgeUid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
printInfos()
{
  info() << "-- Mesh information (Arcane2):";
  info() << "Number of nodes   " << m_one_mesh_item_adder->nbNode();
  info() << "Number of edges   " << m_one_mesh_item_adder->nbEdge();
  info() << "Number of faces   " << m_one_mesh_item_adder->nbFace();
  info() << "Number of cells   " << m_one_mesh_item_adder->nbCell();
  info() << "-- Graph information (Arcane2):";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
printStats(Int32 level)
{
  info(level) << "-- -- Statistics";
  info(level) << "Number of nodes after addition     &: " << m_one_mesh_item_adder->nbNode()
              << " hashmap=" << m_mesh->nodesMap().count();
  info(level) << "Number of edges after addition     : " << m_one_mesh_item_adder->nbEdge()
              << " hashmap=" << m_mesh->edgesMap().count();
  info(level) << "Number of faces after addition     : " << m_one_mesh_item_adder->nbFace()
              << " hashmap=" << m_mesh->facesMap().count();
  info(level) << "Number of cells after addition     : " << m_one_mesh_item_adder->nbCell()
              << " hashmap=" << m_mesh->cellsMap().count();
  info(level) << "--";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calculates the unique IDs for each face.

  The algorithm must work in a way that gives the same face numbering
  sequentially and in parallel regardless of the decomposition, to facilitate
  debugging.

  The numbering principle is as follows: we traverse the cells in increasing
  order of their unique_id, and for each cell, we traverse the list of faces.
  A face is numbered if and only if its backCell() is the current cell or if
  its frontCell() is the current cell but it is a boundary face (nbCell()==1).
  When numbering the faces of the current cell, we first number the faces for
  which it is the backCell(), then the faces for which it is the frontCell().

  This small detail is explained to simplify the determination of numbering in
  the parallel case (todo: explain why...)

  \warning This method should only be called during the initial mesh creation.
*/
void DynamicMeshIncrementalBuilder::
computeFacesUniqueIds()
{
  if (!m_face_unique_id_builder)
    m_face_unique_id_builder = new FaceUniqueIdBuilder(this);
  m_face_unique_id_builder->computeFacesUniqueIds();

  if (m_has_edge) {
    if (!m_edge_unique_id_builder)
      m_edge_unique_id_builder = new EdgeUniqueIdBuilder(this);
    m_edge_unique_id_builder->computeEdgesUniqueIds();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addGhostLayers(bool is_allocate)
{
  debug() << "Add one ghost layer";
  if (!m_ghost_layer_builder)
    m_ghost_layer_builder = new GhostLayerBuilder(m_mesh);
  m_ghost_layer_builder->addGhostLayers(is_allocate);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR
void DynamicMeshIncrementalBuilder::
addGhostChildFromParent(Array<Int64>& ghost_cell_to_refine)
{
  debug() << "Add one ghost layer";
  if (!m_ghost_layer_builder)
    m_ghost_layer_builder = new GhostLayerBuilder(m_mesh);
  m_ghost_layer_builder->addGhostChildFromParent2(ghost_cell_to_refine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_removeNeedRemoveMarkedItems(ItemInternalMap& map, UniqueArray<Int32>& items_local_id)
{
  // Removal of links
  SharedArray<ItemInternal*> items_to_remove;
  items_to_remove.reserve(1000);

  map.eachItem([&](Item item) {
    impl::MutableItemBase mb_item = item.mutableItemBase();
    Integer f = mb_item.flags();
    if (f & ItemFlags::II_NeedRemove) {
      f &= ~ItemFlags::II_NeedRemove;
      mb_item.setFlags(f);
      items_local_id.add(item.localId());
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Removes ghost items.
 *
 Suppresses all items whose owner is not the current sub-domain
 and which do not belong to any internal elements of this sub-domain.
 Internal items that are no longer connected to any items are
 also destroyed
*/
void DynamicMeshIncrementalBuilder::
removeNeedRemoveMarkedItems()
{
  if (!m_mesh->useMeshItemFamilyDependencies()) {

    IItemFamily* links_family = m_mesh->findItemFamily(IK_DoF, GraphDoFs::linkFamilyName(), false, false);
    if (links_family) {
      info() << "Remove Items for family : " << links_family->name();
      links_family->removeNeedRemoveMarkedItems();
    }

    for (IItemFamily* family : m_mesh->itemFamilies()) {
      if (family->itemKind() != IK_DoF || family->name() == GraphDoFs::linkFamilyName())
        continue;
      info() << "Remove Items for family : " << family->name();
      family->removeNeedRemoveMarkedItems();
    }
    // Suppression of particles from the Particle family

    for (IItemFamilyCollection::Enumerator i(m_mesh->itemFamilies()); ++i;) {
      IItemFamily* family = *i;
      if (family->itemKind() == IK_Particle) {
        ParticleFamily* particle_family = dynamic_cast<ParticleFamily*>(family);
        ARCANE_CHECK_POINTER(particle_family);
        if (particle_family && particle_family->getEnableGhostItems() == true) {
          UniqueArray<Integer> lids_to_remove;
          lids_to_remove.reserve(1000);

          ItemInternalMap& particle_map = particle_family->itemsMap();
          particle_map.eachItem([&](Item item) {
            impl::MutableItemBase mb_item = item.mutableItemBase();
            Integer f = mb_item.flags();
            if (f & ItemFlags::II_NeedRemove) {
              f &= ~ItemFlags::II_NeedRemove;
              mb_item.setFlags(f);
              lids_to_remove.add(item.localId());
            }
          });

          info() << "Number of particles of family " << family->name() << " to remove: " << lids_to_remove.size();
          if (lids_to_remove.size() > 0)
            particle_family->removeParticles(lids_to_remove);
        }
      }
    }

    // Suppression of cells
    CellFamily& cell_family = m_mesh->trueCellFamily();
    UniqueArray<Int32> cells_to_remove;
    cells_to_remove.reserve(1000);
    _removeNeedRemoveMarkedItems(cell_family.itemsMap(), cells_to_remove);

    info() << "Number of cells to remove: " << cells_to_remove.size();

#ifdef ARCANE_DEBUG_DYNAMIC_MESH
    for (Integer i = 0, is = cells_to_remove.size(); i < is; ++i)
      info() << "remove cell with uid=" << cells_to_remove[i]->uniqueId()
             << ",lid=" << cells_to_remove[i]->localId()
             << ",owner=" << cells_to_remove[i]->owner();
#endif /* ARCANE_DEBUG_DYNAMIC_MESH */

    cell_family.removeCells(cells_to_remove);
  }
  // With ItemFamilyNetwork
  else {
    // handle families already in the network
    m_mesh->itemFamilyNetwork()->schedule([](IItemFamily* family) {
      family->removeNeedRemoveMarkedItems();
      // Remove items from family graph leaves to root
    },
                                          IItemFamilyNetwork::InverseTopologicalOrder);
    for (auto family : m_mesh->itemFamilies()) {
      // handle remaining families (particles, links, dual nodes & dof).
      // Will have no effect on the families already handled in the graph (since NeedRemove flag is reset)
      family->removeNeedRemoveMarkedItems();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
setConnectivity(const Integer connectivity)
{
  if (connectivity == 0)
    ARCANE_FATAL("Undefined connectivity !");
  if (m_connectivity != 0)
    ARCANE_FATAL("Connectivity already set: cannot redefine it");
  m_connectivity = connectivity;
  m_has_edge = Connectivity::hasConnectivity(connectivity, Connectivity::CT_HasEdge);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
resetAfterDeallocate()
{
  m_face_uid_pool = 0;
  m_edge_uid_pool = 0;
  m_one_mesh_item_adder->resetAfterDeallocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
