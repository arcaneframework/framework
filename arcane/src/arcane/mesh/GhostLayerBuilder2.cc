// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerBuilder2.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Construction of ghost layers.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/parallel/BitonicSortT.H"

#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

#include <algorithm>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction of ghost layers.
 */
class GhostLayerBuilder2
: public TraceAccessor
{
  class BoundaryNodeInfo;
  class BoundaryNodeBitonicSortTraits;
  class BoundaryNodeToSendInfo;

 public:

  using ItemInternalMap = DynamicMeshKindInfos::ItemInternalMap;
  using SubDomainItemMap = HashTableMapT<Int32, SharedArray<Int32>>;

 public:

  //! Constructs an instance for the mesh \a mesh
  GhostLayerBuilder2(DynamicMeshIncrementalBuilder* mesh_builder, bool is_allocate, Int32 version);

 public:

  void addGhostLayers();

 private:

  DynamicMesh* m_mesh = nullptr;
  DynamicMeshIncrementalBuilder* m_mesh_builder = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  bool m_is_verbose = false;
  bool m_is_allocate = false;
  Int32 m_version = -1;
  bool m_use_optimized_node_layer = true;
  bool m_use_only_minimal_cell_uid = true;

 private:

  void _printItem(ItemInternal* ii, std::ostream& o);
  void _markBoundaryItems(ArrayView<Int32> node_layer);
  void _sendAndReceiveCells(SubDomainItemMap& cells_to_send);
  void _sortBoundaryNodeList(Array<BoundaryNodeInfo>& boundary_node_list);
  void _addGhostLayer(Integer current_layer, Int32ConstArrayView node_layer);
  void _markBoundaryNodes(ArrayView<Int32> node_layer);
  void _markBoundaryNodesFromEdges(ArrayView<Int32> node_layer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerBuilder2::
GhostLayerBuilder2(DynamicMeshIncrementalBuilder* mesh_builder, bool is_allocate, Int32 version)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
, m_parallel_mng(m_mesh->parallelMng())
, m_is_allocate(is_allocate)
, m_version(version)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_GHOSTLAYER_USE_OPTIMIZED_LAYER", true)) {
    Int32 vv = v.value();
    m_use_optimized_node_layer = (vv == 1 || vv == 3);
    m_use_only_minimal_cell_uid = (v == 2 || vv == 3);
  }
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_GHOSTLAYER_VERBOSE", true)) {
    m_is_verbose = (v.value() != 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_printItem(ItemInternal* ii, std::ostream& o)
{
  o << ItemPrinter(ii);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure containing boundary node information.
 *
 * This structure is used to communicate with other ranks.
 * Therefore, it must be a POD type. For communication with other
 * ranks, it must be converted into a base type which is an Int64.
 * Therefore, its size must also be a multiple of that of an Int64.
 */
class GhostLayerBuilder2::BoundaryNodeInfo
{
 public:

  using BasicType = Int64;
  static constexpr Int64 nbBasicTypeSize() { return 3; }

 public:

  static ConstArrayView<BasicType> asBasicBuffer(ConstArrayView<BoundaryNodeInfo> values)
  {
    Int32 message_size = messageSize(values);
    const BoundaryNodeInfo* fsi_base = values.data();
    auto* ptr = reinterpret_cast<const Int64*>(fsi_base);
    return ConstArrayView<BasicType>(message_size, ptr);
  }

  static ArrayView<BasicType> asBasicBuffer(ArrayView<BoundaryNodeInfo> values)
  {
    Int32 message_size = messageSize(values);
    BoundaryNodeInfo* fsi_base = values.data();
    auto* ptr = reinterpret_cast<Int64*>(fsi_base);
    return ArrayView<BasicType>(message_size, ptr);
  }

  static Int32 messageSize(ConstArrayView<BoundaryNodeInfo> values)
  {
    static_assert((sizeof(Int64) * nbBasicTypeSize()) == sizeof(BoundaryNodeInfo));
    Int64 message_size_i64 = values.size() * nbBasicTypeSize();
    Int32 message_size = CheckedConvert::toInteger(message_size_i64);
    return message_size;
  }

  static Int32 nbElement(Int32 message_size)
  {
    if ((message_size % nbBasicTypeSize()) != 0)
      ARCANE_FATAL("Message size '{0}' is not a multiple of basic size '{1}'", message_size, nbBasicTypeSize());
    Int32 nb_element = message_size / nbBasicTypeSize();
    return nb_element;
  }

 public:

  struct HashFunction
  {
    size_t operator()(const BoundaryNodeInfo& a) const
    {
      size_t h1 = std::hash<Int64>{}(a.node_uid);
      size_t h2 = std::hash<Int64>{}(a.cell_uid);
      size_t h3 = std::hash<Int32>{}(a.cell_owner);
      return h1 ^ h2 ^ h3;
    }
  };
  friend bool operator==(const BoundaryNodeInfo& a, const BoundaryNodeInfo& b)
  {
    return (a.node_uid == b.node_uid && a.cell_uid == b.cell_uid && a.cell_owner == b.cell_owner);
  }

 public:

  Int64 node_uid = NULL_ITEM_UNIQUE_ID;
  Int64 cell_uid = NULL_ITEM_UNIQUE_ID;
  Int32 cell_owner = -1;
  Int32 padding = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Functor for sorting BoundaryNodeInfo via bitonic sort.
 */
class GhostLayerBuilder2::BoundaryNodeBitonicSortTraits
{
 public:

  static bool compareLess(const BoundaryNodeInfo& k1, const BoundaryNodeInfo& k2)
  {
    Int64 k1_node_uid = k1.node_uid;
    Int64 k2_node_uid = k2.node_uid;
    if (k1_node_uid < k2_node_uid)
      return true;
    if (k1_node_uid > k2_node_uid)
      return false;

    Int64 k1_cell_uid = k1.cell_uid;
    Int64 k2_cell_uid = k2.cell_uid;
    if (k1_cell_uid < k2_cell_uid)
      return true;
    if (k1_cell_uid > k2_cell_uid)
      return false;

    return (k1.cell_owner < k2.cell_owner);
  }

  static Parallel::Request send(IParallelMng* pm, Int32 rank, ConstArrayView<BoundaryNodeInfo> values)
  {
    auto buf_view = BoundaryNodeInfo::asBasicBuffer(values);
    return pm->send(buf_view, rank, false);
  }

  static Parallel::Request recv(IParallelMng* pm, Int32 rank, ArrayView<BoundaryNodeInfo> values)
  {
    auto buf_view = BoundaryNodeInfo::asBasicBuffer(values);
    return pm->recv(buf_view, rank, false);
  }

  static Integer messageSize(ConstArrayView<BoundaryNodeInfo> values)
  {
    return BoundaryNodeInfo::messageSize(values);
  }

  static BoundaryNodeInfo maxValue()
  {
    BoundaryNodeInfo bni;
    bni.node_uid = INT64_MAX;
    bni.cell_uid = INT64_MAX;
    bni.cell_owner = -1;
    return bni;
  }

  static bool isValid(const BoundaryNodeInfo& bni)
  {
    return bni.node_uid != INT64_MAX;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GhostLayerBuilder2::BoundaryNodeToSendInfo
{
 public:

  Integer m_index;
  Integer m_nb_cell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds ghost cell layers.
 *
 * This version uses sorting to determine the info
 *
 * Before calling this function, there should be no ghost cells: all cells
 * in the mesh must belong to this sub-domain.
 * (TODO: add test for this).
 *
 * If multiple ghost cell layers are requested, we proceed in multiple
 * steps for the same algorithm. First, we send the first layer, then the first
 * and the second, then three layers, and so on. This is probably not
 * optimal in terms of communication but allows processing all cases,
 * especially the case where multiple sub-domains must be crossed to
 * add ghost cell layers.
 *
 * \todo: implement the optimizations specified in the comments
 * in this function.
 * \todo: ensure that we only work with the cell/node connectivity.
 *
 */
void GhostLayerBuilder2::
addGhostLayers()
{
  IParallelMng* pm = m_parallel_mng;
  if (!pm->isParallel())
    return;
  Integer nb_ghost_layer = m_mesh->ghostLayerMng()->nbGhostLayer();
  info() << "** GHOST LAYER BUILDER V" << m_version << " with sort (nb_ghost_layer=" << nb_ghost_layer << ")";

  // Ghost layer to which the node belongs.
  UniqueArray<Integer> node_layer(m_mesh->nodeFamily()->maxLocalId(), -1);

  // Mark boundary items
  // We do this even if we do not want ghost cell layers
  _markBoundaryItems(node_layer);

  if (nb_ghost_layer == 0)
    return;
  const Int32 my_rank = pm->commRank();

  const bool is_non_manifold = m_mesh->meshKind().isNonManifold();
  if (is_non_manifold && (m_version != 3))
    ARCANE_FATAL("Only version 3 of ghostlayer builder is supported for non manifold meshes");

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  Integer boundary_nodes_uid_count = 0;

  // Check that there are no ghost cells with version 3.
  // If so, display a warning and indicate to use version 4.
  if (m_version == 3) {
    Integer nb_ghost = 0;
    cells_map.eachItem([&](Item cell) {
      if (!cell.isOwn())
        ++nb_ghost;
    });
    if (nb_ghost != 0)
      warning() << "Invalid call to addGhostLayers() with version 3 because mesh "
                << " already has '" << nb_ghost << "' ghost cells. The computed ghost cells"
                << " may be wrong. Use version 4 of ghost layer builder if you want to handle this case";
  }

  // Ghost layer to which the cell belongs.
  UniqueArray<Integer> cell_layer(m_mesh->cellFamily()->maxLocalId(), -1);

  if (m_version >= 4) {
    _markBoundaryNodes(node_layer);
    nodes_map.eachItem([&](Item node) {
      if (node_layer[node.localId()] == 1)
        ++boundary_nodes_uid_count;
    });
  }
  else {
    // Iterate over nodes and calculate the number of boundary nodes
    // and marks the first layer
    nodes_map.eachItem([&](Item node) {
      Int32 f = node.itemBase().flags();
      if (f & ItemFlags::II_Shared) {
        node_layer[node.localId()] = 1;
        ++boundary_nodes_uid_count;
      }
    });
  }

  info() << "NB BOUNDARY NODE=" << boundary_nodes_uid_count;

  for (Integer current_layer = 1; current_layer <= nb_ghost_layer; ++current_layer) {
    //Integer current_layer = 1;
    info() << "Processing layer " << current_layer;
    cells_map.eachItem([&](Cell cell) {
      // Do not process cells that do not belong to me
      if (m_version >= 4 && cell.owner() != my_rank)
        return;
      //Int64 cell_uid = cell->uniqueId();
      Int32 cell_lid = cell.localId();
      if (cell_layer[cell_lid] != (-1))
        return;
      bool is_current_layer = false;
      for (Int32 inode_local_id : cell.nodeIds()) {
        Integer layer = node_layer[inode_local_id];
        //info() << "NODE_LAYER lid=" << i_node->localId() << " layer=" << layer;
        if (layer == current_layer) {
          is_current_layer = true;
          break;
        }
      }
      if (is_current_layer) {
        cell_layer[cell_lid] = current_layer;
        //info() << "Current layer celluid=" << cell_uid;
        // If not marked, initialize to the current layer + 1.
        for (Int32 inode_local_id : cell.nodeIds()) {
          Integer layer = node_layer[inode_local_id];
          if (layer == (-1)) {
            //info() << "Marks node uid=" << i_node->uniqueId();
            node_layer[inode_local_id] = current_layer + 1;
          }
        }
      }
    });
  }

  // Marks the nodes for which the ghost layer has not yet been assigned.
  // For them, we indicate that we are on layer 'nb_ghost_layer+1'.
  // The goal is never to transfer these nodes.
  // NOTE: This mechanism was added in July 2024 for version 3.14.
  //       If it works well, we might only keep this method.
  if (m_use_optimized_node_layer) {
    Integer nb_no_layer = 0;
    nodes_map.eachItem([&](Node node) {
      Int32 lid = node.localId();
      Int32 layer = node_layer[lid];
      if (layer <= 0) {
        node_layer[lid] = nb_ghost_layer + 1;
        ++nb_no_layer;
      }
    });
    info() << "Mark remaining nodes nb=" << nb_no_layer;
  }

  for (Integer i = 1; i <= nb_ghost_layer; ++i)
    _addGhostLayer(i, node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Determines the boundary nodes.
 *
 * This algorithm works even if there are already ghost cells.
 * To determine the boundary nodes, you must first determine the
 * boundary faces. A face is boundary if it falls into one of two cases:
 * - it has only one connected cell belonging to this subdomain.
 * - it is connected to two cells, exactly one of which belongs to this
 *   domain.
 * Once the boundary faces are found, we consider that the boundary nodes
 * are those that belong to a boundary face.
 */
void GhostLayerBuilder2::
_markBoundaryNodes(ArrayView<Int32> node_layer)
{
  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();
  // TODO: check if it is correct to modify ItemFlags::II_SubDomainBoundary
  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;
  // Iterates over faces and marks boundary nodes, edges and faces
  faces_map.eachItem([&](Face face) {
    Int32 nb_own = 0;
    for (Integer i = 0, n = face.nbCell(); i < n; ++i)
      if (face.cell(i).owner() == my_rank)
        ++nb_own;
    if (nb_own == 1) {
      face.mutableItemBase().addFlags(shared_and_boundary_flags);
      //++nb_sub_domain_boundary_face;
      for (Item inode : face.nodes()) {
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
        node_layer[inode.localId()] = 1;
      }
      for (Item iedge : face.edges())
        iedge.mutableItemBase().addFlags(shared_and_boundary_flags);
    }
  });
  _markBoundaryNodesFromEdges(node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_addGhostLayer(Integer current_layer, Int32ConstArrayView node_layer)
{
  info() << "Processing ghost layer " << current_layer;

  SharedArray<BoundaryNodeInfo> boundary_node_list;
  //boundary_node_list.reserve(boundary_nodes_uid_count);

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  Int64 nb_added_for_different_rank = 0;
  Int64 nb_added_for_in_layer = 0;

  const Int32 max_local_id = m_mesh->nodeFamily()->maxLocalId();

  // Arrays containing for each node the uid of the smallest connected cell
  // and the associated rank. If the uid is A_NULL_UNIQUE_ID, this node should not be added.
  UniqueArray<Int64> node_cell_uids(max_local_id, NULL_ITEM_UNIQUE_ID);

  const bool do_only_minimal_uid = m_use_only_minimal_cell_uid;
  // We must send all nodes whose layer number is different from (-1).
  // NOTE: for the layer above 1, only one value must be sent.
  cells_map.eachItem([&](Cell cell) {
    // Do not process cells that do not belong to me
    if (m_version >= 4 && cell.owner() != my_rank)
      return;
    Int64 cell_uid = cell.uniqueId();
    for (Node node : cell.nodes()) {
      Int32 node_lid = node.localId();
      bool do_it = false;
      if (cell.owner() != my_rank) {
        do_it = true;
        ++nb_added_for_different_rank;
      }
      else {
        Integer layer = node_layer[node_lid];
        do_it = layer <= current_layer;
        if (do_it)
          ++nb_added_for_in_layer;
      }
      if (do_it) {
        Int32 node_lid = node.localId();
        if (do_only_minimal_uid) {
          Int64 current_uid = node_cell_uids[node_lid];
          if ((current_uid == NULL_ITEM_UNIQUE_ID) || cell_uid < current_uid) {
            node_cell_uids[node_lid] = cell_uid;
            if (is_verbose)
              info() << "AddNode node_uid=" << node.uniqueId() << " cell=" << cell_uid;
          }
          else if (is_verbose)
            info() << "AddNode node_uid=" << node.uniqueId() << " cell=" << cell_uid << " not done current=" << current_uid;
        }
        else {
          Int64 node_uid = node.uniqueId();
          BoundaryNodeInfo nci;
          nci.node_uid = node_uid;
          nci.cell_uid = cell_uid;
          nci.cell_owner = my_rank;
          boundary_node_list.add(nci);
          if (is_verbose)
            info() << "AddNode node_uid=" << node.uniqueId() << " cell=" << cell_uid;
        }
      }
    }
  });

  if (do_only_minimal_uid) {
    nodes_map.eachItem([&](Node node) {
      Int32 lid = node.localId();
      Int64 cell_uid = node_cell_uids[lid];
      if (cell_uid != NULL_ITEM_UNIQUE_ID) {
        Int64 node_uid = node.uniqueId();
        BoundaryNodeInfo nci;
        nci.node_uid = node_uid;
        nci.cell_uid = cell_uid;
        nci.cell_owner = my_rank;
        boundary_node_list.add(nci);
      }
    });
  }

  info() << "NB BOUNDARY NODE LIST=" << boundary_node_list.size()
         << " nb_added_for_different_rank=" << nb_added_for_different_rank
         << " nb_added_for_in_layer=" << nb_added_for_in_layer
         << " do_only_minimal=" << do_only_minimal_uid;

  _sortBoundaryNodeList(boundary_node_list);
  SharedArray<BoundaryNodeInfo> all_boundary_node_info = boundary_node_list;

  UniqueArray<BoundaryNodeToSendInfo> node_list_to_send;
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer bi_n = all_bni.size();
    for (Integer i = 0; i < bi_n; ++i) {
      const BoundaryNodeInfo& bni = all_bni[i];
      // Searches all elements of all_bni that have the same node.
      // This represents all cells connected to this node.
      Int64 node_uid = bni.node_uid;
      Integer last_i = i;
      for (; last_i < bi_n; ++last_i)
        if (all_bni[last_i].node_uid != node_uid)
          break;
      Integer nb_same_node = (last_i - i);
      if (is_verbose)
        info() << "NB_SAME_NODE uid=" << node_uid << " n=" << nb_same_node << " last_i=" << last_i;
      // Now, check if the cells connected to this node have the same owner.
      // If this is the case, it is a true boundary node and nothing needs to be done.
      // Otherwise, the list of cells must be sent to all PEs whose ranks appear in this list
      Int32 owner = bni.cell_owner;
      bool has_ghost = false;
      for (Integer z = 0; z < nb_same_node; ++z)
        if (all_bni[i + z].cell_owner != owner) {
          has_ghost = true;
          break;
        }
      if (has_ghost) {
        BoundaryNodeToSendInfo si;
        si.m_index = i;
        si.m_nb_cell = nb_same_node;
        node_list_to_send.add(si);
        if (is_verbose)
          info() << "Add ghost uid=" << node_uid << " index=" << i << " nb_same_node=" << nb_same_node;
      }
      i = last_i - 1;
    }
  }

  IntegerUniqueArray nb_info_to_send(nb_rank, 0);
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer nb_node_to_send = node_list_to_send.size();
    std::set<Int32> ranks_done;
    for (Integer i = 0; i < nb_node_to_send; ++i) {
      Integer index = node_list_to_send[i].m_index;
      Integer nb_cell = node_list_to_send[i].m_nb_cell;

      ranks_done.clear();

      for (Integer kz = 0; kz < nb_cell; ++kz) {
        Int32 krank = all_bni[index + kz].cell_owner;
        if (ranks_done.find(krank) == ranks_done.end()) {
          ranks_done.insert(krank);
          // For each one, it will be necessary to send
          // - the number of cells (1*Int64)
          // - the node uid (1*Int64)
          // - the uid and rank of each cell (2*Int64*nb_cell)
          //TODO: it is possible to store the ranks as Int32
          nb_info_to_send[krank] += (nb_cell * 2) + 2;
        }
      }
    }
  }

  if (is_verbose) {
    for (Integer i = 0; i < nb_rank; ++i) {
      Integer nb_to_send = nb_info_to_send[i];
      if (nb_to_send != 0)
        info() << "NB_TO_SEND rank=" << i << " n=" << nb_to_send;
    }
  }

  Integer total_nb_to_send = 0;
  IntegerUniqueArray nb_info_to_send_indexes(nb_rank, 0);
  for (Integer i = 0; i < nb_rank; ++i) {
    nb_info_to_send_indexes[i] = total_nb_to_send;
    total_nb_to_send += nb_info_to_send[i];
  }
  info() << "TOTAL_NB_TO_SEND=" << total_nb_to_send;

  UniqueArray<Int64> resend_infos(total_nb_to_send);
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer nb_node_to_send = node_list_to_send.size();
    std::set<Int32> ranks_done;
    for (Integer i = 0; i < nb_node_to_send; ++i) {
      Integer node_index = node_list_to_send[i].m_index;
      Integer nb_cell = node_list_to_send[i].m_nb_cell;
      Int64 node_uid = all_bni[node_index].node_uid;

      ranks_done.clear();

      for (Integer kz = 0; kz < nb_cell; ++kz) {
        Int32 krank = all_bni[node_index + kz].cell_owner;
        if (ranks_done.find(krank) == ranks_done.end()) {
          ranks_done.insert(krank);
          Integer send_index = nb_info_to_send_indexes[krank];
          resend_infos[send_index] = node_uid;
          ++send_index;
          resend_infos[send_index] = nb_cell;
          ++send_index;
          for (Integer zz = 0; zz < nb_cell; ++zz) {
            resend_infos[send_index] = all_bni[node_index + zz].cell_uid;
            ++send_index;
            resend_infos[send_index] = all_bni[node_index + zz].cell_owner;
            ++send_index;
          }
          nb_info_to_send_indexes[krank] = send_index;
        }
      }
    }
  }

  IntegerUniqueArray nb_info_to_recv(nb_rank, 0);
  {
    Timer::SimplePrinter sp(traceMng(), "Sending size with AllToAll");
    pm->allToAll(nb_info_to_send, nb_info_to_recv, 1);
  }

  if (is_verbose)
    for (Integer i = 0; i < nb_rank; ++i)
      info() << "NB_TO_RECV: I=" << i << " n=" << nb_info_to_recv[i];

  Integer total_nb_to_recv = 0;
  for (Integer i = 0; i < nb_rank; ++i)
    total_nb_to_recv += nb_info_to_recv[i];

  // There is a high chance that this will not work if the array is too large,
  // one must proceed with arrays that do not exceed 2Go because of the
  // MPI Int32.
  // TODO: Perform the AllToAll in several stages if necessary.
  // TODO: Merge this code with that of FaceUniqueIdBuilder2.
  UniqueArray<Int64> recv_infos;
  {
    Int32 vsize = sizeof(Int64) / sizeof(Int64);
    Int32UniqueArray send_counts(nb_rank);
    Int32UniqueArray send_indexes(nb_rank);
    Int32UniqueArray recv_counts(nb_rank);
    Int32UniqueArray recv_indexes(nb_rank);
    Int32 total_send = 0;
    Int32 total_recv = 0;
    for (Integer i = 0; i < nb_rank; ++i) {
      send_counts[i] = (Int32)(nb_info_to_send[i] * vsize);
      recv_counts[i] = (Int32)(nb_info_to_recv[i] * vsize);
      send_indexes[i] = total_send;
      recv_indexes[i] = total_recv;
      total_send += send_counts[i];
      total_recv += recv_counts[i];
    }
    recv_infos.resize(total_nb_to_recv);

    Int64ConstArrayView send_buf(total_nb_to_send * vsize, (Int64*)resend_infos.data());
    Int64ArrayView recv_buf(total_nb_to_recv * vsize, (Int64*)recv_infos.data());

    info() << "BUF_SIZES: send=" << send_buf.size() << " recv=" << recv_buf.size();
    {
      Timer::SimplePrinter sp(traceMng(), "Send values with AllToAll");
      pm->allToAllVariable(send_buf, send_counts, send_indexes, recv_buf, recv_counts, recv_indexes);
    }
  }

  SubDomainItemMap cells_to_send(50, true);

  // TODO: we don't necessarily need the cells here, but
  // only the list of procs to whom it must be sent. Then,
  // if the proc knows to whom it must send, it can send the cells
  // at that time. This allows sending less information in the previous AllToAll.

  {
    Integer index = 0;
    UniqueArray<Int32> my_cells;
    SharedArray<Int32> ranks_to_send;
    std::set<Int32> ranks_done;
    while (index < total_nb_to_recv) {
      Int64 node_uid = recv_infos[index];
      ++index;
      Int64 nb_cell = recv_infos[index];
      ++index;
      Node current_node(nodes_map.findItem(node_uid));
      if (is_verbose)
        info() << "NODE uid=" << node_uid << " nb_cell=" << nb_cell << " idx=" << (index - 2);
      my_cells.clear();
      ranks_to_send.clear();
      ranks_done.clear();
      for (Integer kk = 0; kk < nb_cell; ++kk) {
        Int64 cell_uid = recv_infos[index];
        ++index;
        Int32 cell_owner = CheckedConvert::toInt32(recv_infos[index]);
        ++index;
        if (kk == 0 && current_layer == 1 && m_is_allocate)
          // I am the cell with the smallest uid and therefore I
          // position the node owner.
          // TODO: do not do this here, but do it in a separate routine.
          nodes_map.findItem(node_uid).toMutable().setOwner(cell_owner, my_rank);
        if (is_verbose)
          info() << " CELL=" << cell_uid << " owner=" << cell_owner;
        if (cell_owner == my_rank) {
          impl::ItemBase dcell = cells_map.tryFind(cell_uid);
          if (dcell.null())
            ARCANE_FATAL("Internal error: cell uid={0} is not in our mesh", cell_uid);
          if (do_only_minimal_uid) {
            // Add all cells around my node
            for (CellLocalId c : current_node.cellIds())
              my_cells.add(c);
          }
          else
            my_cells.add(dcell.localId());
        }
        else {
          if (ranks_done.find(cell_owner) == ranks_done.end()) {
            ranks_to_send.add(cell_owner);
            ranks_done.insert(cell_owner);
          }
        }
      }

      if (is_verbose) {
        info() << "CELLS TO SEND: node_uid=" << node_uid
               << " nb_rank=" << ranks_to_send.size()
               << " nb_cell=" << my_cells.size();
        info(4) << "CELLS TO SEND: node_uid=" << node_uid
                << " rank=" << ranks_to_send
                << " cell=" << my_cells;
      }

      for (Integer zrank = 0, zn = ranks_to_send.size(); zrank < zn; ++zrank) {
        Int32 send_rank = ranks_to_send[zrank];
        SubDomainItemMap::Data* d = cells_to_send.lookupAdd(send_rank);
        Int32Array& c = d->value();
        for (Integer zid = 0, zid_size = my_cells.size(); zid < zid_size; ++zid) {
          // TODO: check if cell is already present and do not add it if it is not necessary.
          c.add(my_cells[zid]);
        }
      }
    }
  }

  info() << "GHOST V3 SERIALIZE CELLS";
  _sendAndReceiveCells(cells_to_send);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Parallel sorting of the list of boundary node information.
 *
 * Takes as input a list of boundary nodes and sorts it in parallel
 * ensuring that after sorting the information for the same node is on the
 * same process.
 */
void GhostLayerBuilder2::
_sortBoundaryNodeList(Array<BoundaryNodeInfo>& boundary_node_list)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  bool is_verbose = m_is_verbose;

  Parallel::BitonicSort<BoundaryNodeInfo, BoundaryNodeBitonicSortTraits> boundary_node_sorter(pm);
  boundary_node_sorter.setNeedIndexAndRank(false);

  {
    Timer::SimplePrinter sp(traceMng(), "Sorting boundary nodes");
    boundary_node_sorter.sort(boundary_node_list);
  }

  if (is_verbose) {
    ConstArrayView<BoundaryNodeInfo> all_bni = boundary_node_sorter.keys();
    Integer n = all_bni.size();
    for (Integer i = 0; i < n; ++i) {
      const BoundaryNodeInfo& bni = all_bni[i];
      info() << "NODES_KEY i=" << i
             << " node=" << bni.node_uid
             << " cell=" << bni.cell_uid
             << " rank=" << bni.cell_owner;
    }
  }

  // TODO: it is not necessary to send all the cells.
  // to determine the owner of a node, it is sufficient
  // for each PE to send its cell with the smallest UID.
  // Then, each node needs to know the list
  // of connected sub-domains to send the info. Each
  // sub-domain, knowing this, will know to whom it must send
  // the ghost cells.

  {
    ConstArrayView<BoundaryNodeInfo> all_bni = boundary_node_sorter.keys();
    Integer n = all_bni.size();
    // Since the same node may be present in the list of the previous proc, each PE
    // (except 0) sends to the previous proc the beginning of its list which contains the same nodes.

    UniqueArray<BoundaryNodeInfo> end_node_list;
    Integer begin_own_list_index = 0;
    if (n != 0 && my_rank != 0) {
      if (BoundaryNodeBitonicSortTraits::isValid(all_bni[0])) {
        Int64 node_uid = all_bni[0].node_uid;
        for (Integer i = 0; i < n; ++i) {
          if (all_bni[i].node_uid != node_uid) {
            begin_own_list_index = i;
            break;
          }
          else
            end_node_list.add(all_bni[i]);
        }
      }
    }
    info() << "BEGIN_OWN_LIST_INDEX=" << begin_own_list_index << " end_node_list_size=" << end_node_list.size();
    if (is_verbose) {
      for (Integer k = 0, kn = end_node_list.size(); k < kn; ++k)
        info() << " SEND node_uid=" << end_node_list[k].node_uid
               << " cell_uid=" << end_node_list[k].cell_uid;
    }

    UniqueArray<BoundaryNodeInfo> end_node_list_recv;

    UniqueArray<Parallel::Request> requests;
    Integer recv_message_size = 0;
    Integer send_message_size = BoundaryNodeBitonicSortTraits::messageSize(end_node_list);

    // Send and receive sizes first.
    if (my_rank != (nb_rank - 1)) {
      requests.add(pm->recv(IntegerArrayView(1, &recv_message_size), my_rank + 1, false));
    }
    if (my_rank != 0) {
      requests.add(pm->send(IntegerConstArrayView(1, &send_message_size), my_rank - 1, false));
    }
    info() << "Send size=" << send_message_size << " Recv size=" << recv_message_size;
    pm->waitAllRequests(requests);
    requests.clear();

    if (recv_message_size != 0) {
      Int32 nb_element = BoundaryNodeInfo::nbElement(recv_message_size);
      end_node_list_recv.resize(nb_element);
      requests.add(BoundaryNodeBitonicSortTraits::recv(pm, my_rank + 1, end_node_list_recv));
    }
    if (send_message_size != 0)
      requests.add(BoundaryNodeBitonicSortTraits::send(pm, my_rank - 1, end_node_list));

    pm->waitAllRequests(requests);

    boundary_node_list.clear();
    boundary_node_list.addRange(all_bni.subConstView(begin_own_list_index, n - begin_own_list_index));
    boundary_node_list.addRange(end_node_list_recv);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_sendAndReceiveCells(SubDomainItemMap& cells_to_send)
{
  auto exchanger{ ParallelMngUtils::createExchangerRef(m_parallel_mng) };

  const bool is_verbose = m_is_verbose;

  // Envoie et réceptionne les mailles fantômes
  for (SubDomainItemMap::Enumerator i_map(cells_to_send); ++i_map;) {
    Int32 sd = i_map.data()->key();
    Int32Array& items = i_map.data()->value();

    // Comme la liste par sous-domaine peut contenir plusieurs
    // fois la même maille, on trie la liste et on supprime les
    // doublons
    std::sort(std::begin(items), std::end(items));
    auto new_end = std::unique(std::begin(items), std::end(items));
    items.resize(CheckedConvert::toInteger(new_end - std::begin(items)));
    if (is_verbose)
      info(4) << "CELLS TO SEND SD=" << sd << " Items=" << items;
    else
      info(4) << "CELLS TO SEND SD=" << sd << " nb=" << items.size();
    exchanger->addSender(sd);
  }
  exchanger->initializeCommunicationsMessages();
  for (Integer i = 0, ns = exchanger->nbSender(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToSend(i);
    Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    Int32ConstArrayView items_to_send = cells_to_send[rank];
    m_mesh->serializeCells(s, items_to_send);
  }
  exchanger->processExchange();
  info(4) << "END EXCHANGE CELLS";
  for (Integer i = 0, ns = exchanger->nbReceiver(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    m_mesh->addCells(s);
  }
  m_mesh_builder->printStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Marks the entities at the edge of the sub-domain.
 *
 * This assumes that the faces have already been marked with the II_Boundary flag
 * and that their owner is correctly positioned (i.e.: the same for
 * all sub-domains).
 */
void GhostLayerBuilder2::
_markBoundaryItems(ArrayView<Int32> node_layer)
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();

  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;

  // Iterates over all faces and marks boundary nodes, edges and faces
  faces_map.eachItem([&](Face face) {
    bool is_sub_domain_boundary_face = false;
    if (face.itemBase().flags() & ItemFlags::II_Boundary) {
      is_sub_domain_boundary_face = true;
    }
    else {
      if (face.nbCell() == 2 && (face.cell(0).owner() != my_rank || face.cell(1).owner() != my_rank))
        is_sub_domain_boundary_face = true;
    }
    if (is_sub_domain_boundary_face) {
      face.mutableItemBase().addFlags(shared_and_boundary_flags);
      for (Item inode : face.nodes())
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
      for (Item iedge : face.edges())
        iedge.mutableItemBase().addFlags(shared_and_boundary_flags);
    }
  });
  _markBoundaryNodesFromEdges(node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_markBoundaryNodesFromEdges(ArrayView<Int32> node_layer)
{
  const bool is_non_manifold = m_mesh->meshKind().isNonManifold();
  if (!is_non_manifold)
    return;

  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;

  info() << "Mark boundary nodes from edges for non-manifold mesh";
  // Iterates over all edges.
  // If an edge is connected to only one 2D cell
  // whose owner we are, then it is a boundary edge
  // and we mark the corresponding nodes.
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  ItemInternalMap& edges_map = m_mesh->edgesMap();
  edges_map.eachItem([&](Edge edge) {
    Int32 nb_cell = edge.nbCell();
    Int32 nb_dim2_cell = 0;
    Int32 nb_own_dim2_cell = 0;
    for (Cell cell : edge.cells()) {
      Int32 dim = cell.typeInfo()->dimension();
      if (dim == 2) {
        ++nb_dim2_cell;
        if (cell.owner() == my_rank)
          ++nb_own_dim2_cell;
      }
    }
    if (nb_dim2_cell == nb_cell && nb_own_dim2_cell == 1) {
      edge.mutableItemBase().addFlags(shared_and_boundary_flags);
      for (Item inode : edge.nodes()) {
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
        node_layer[inode.localId()] = 1;
      }
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// This function handles versions 3 and 4 of ghost entity calculation.
extern "C++" void
_buildGhostLayerNewVersion(DynamicMesh* mesh, bool is_allocate, Int32 version)
{
  GhostLayerBuilder2 glb(mesh->m_mesh_builder, is_allocate, version);
  glb.addGhostLayers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
