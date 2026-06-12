// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeDirectionMng.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Information about the cells in an X, Y, or Z direction of a structured    */
/* mesh.                                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/NodeDirectionMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeDirectionMng::Impl
{
 public:

  Impl()
  : m_infos(platform::getDefaultDataAllocator())
  {}

 public:

  NodeGroup m_inner_all_items;
  NodeGroup m_outer_all_items;
  NodeGroup m_inpatch_all_items;
  NodeGroup m_overlap_all_items;
  NodeGroup m_all_items;
  ICartesianMesh* m_cartesian_mesh = nullptr;
  Integer m_patch_index = -1;
  UniqueArray<ItemDirectionInfo> m_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeDirectionMng::
NodeDirectionMng()
: m_direction(MD_DirInvalid)
, m_p(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_internalInit(ICartesianMesh* cm, eMeshDirection dir, Integer patch_index)
{
  if (m_p)
    ARCANE_FATAL("Initialisation already done");
  m_p = new Impl();
  m_direction = dir;
  m_nodes = NodeInfoListView(cm->mesh()->nodeFamily());
  m_p->m_cartesian_mesh = cm;
  m_p->m_patch_index = patch_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_internalDestroy()
{
  delete m_p;
  m_p = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_internalResizeInfos(Int32 new_size)
{
  m_p->m_infos.resize(new_size);
  m_infos_view = m_p->m_infos.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_internalComputeInfos(const CellDirectionMng& cell_dm, const NodeGroup& all_nodes,
                      const VariableCellReal3& cells_center)
{
  Node null_node;
  m_infos_view.fill(NodeDirectionMng::ItemDirectionInfo());

  Integer mesh_dim = m_p->m_cartesian_mesh->mesh()->dimension();
  //TODO: ne garder que les noeuds de notre patch

  // Calcul les infos de direction pour les noeuds
  ENUMERATE_CELL (icell, cell_dm.allCells()) {
    Cell cell = *icell;
    DirCellNode cn(cell_dm.cellNode(cell));

    NodeLocalId node_next_left = cn.nextLeftId();
    NodeLocalId node_next_right = cn.nextRightId();

    NodeLocalId node_previous_left = cn.previousLeftId();
    NodeLocalId node_previous_right = cn.previousRightId();

    m_infos_view[node_previous_left].m_next_lid = node_next_left;
    m_infos_view[node_next_left].m_previous_lid = node_previous_left;

    m_infos_view[node_previous_right].m_next_lid = node_next_right;
    m_infos_view[node_next_right].m_previous_lid = node_previous_right;

    if (mesh_dim == 3) {
      NodeLocalId top_node_next_left = cn.topNextLeftId();
      NodeLocalId top_node_next_right = cn.topNextRightId();

      NodeLocalId top_node_previous_left = cn.topPreviousLeftId();
      NodeLocalId top_node_previous_right = cn.topPreviousRightId();

      m_infos_view[top_node_previous_left].m_next_lid = top_node_next_left;
      m_infos_view[top_node_next_left].m_previous_lid = top_node_previous_left;

      m_infos_view[top_node_previous_right].m_next_lid = top_node_next_right;
      m_infos_view[top_node_next_right].m_previous_lid = top_node_previous_right;
    }
  }

  Int32UniqueArray inner_lids;
  Int32UniqueArray outer_lids;
  IItemFamily* family = all_nodes.itemFamily();
  ENUMERATE_ITEM (iitem, all_nodes) {
    Int32 lid = iitem.itemLocalId();
    Int32 i1 = m_infos_view[lid].m_next_lid;
    Int32 i2 = m_infos_view[lid].m_previous_lid;
    if (i1 == NULL_ITEM_LOCAL_ID || i2 == NULL_ITEM_LOCAL_ID)
      outer_lids.add(lid);
    else
      inner_lids.add(lid);
  }
  int dir = (int)m_direction;
  String base_group_name = String("Direction") + dir;
  if (m_p->m_patch_index >= 0)
    base_group_name = base_group_name + String("AMRPatch") + m_p->m_patch_index;
  m_p->m_inner_all_items = family->createGroup(String("AllInner") + base_group_name, inner_lids, true);
  m_p->m_outer_all_items = family->createGroup(String("AllOuter") + base_group_name, outer_lids, true);
  m_p->m_all_items = all_nodes;

  _filterNodes();
  _computeNodeCellInfos(cell_dm, cells_center);

  {
    UnstructuredMeshConnectivityView mesh_connectivity;
    mesh_connectivity.setMesh(m_p->m_cartesian_mesh->mesh());
    m_node_cell_view = mesh_connectivity.nodeCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_internalComputeInfos(const CellDirectionMng& cell_dm, const NodeGroup& all_nodes)
{
  m_infos_view.fill(ItemDirectionInfo());

  Integer mesh_dim = m_p->m_cartesian_mesh->mesh()->dimension();
  //TODO: ne garder que les noeuds de notre patch

  // Calcul les infos de direction pour les noeuds
  ENUMERATE_CELL (icell, cell_dm.allCells()) {
    Cell cell = *icell;
    DirCellNode cn(cell_dm.cellNode(cell));

    NodeLocalId node_next_left = cn.nextLeftId();
    NodeLocalId node_next_right = cn.nextRightId();

    NodeLocalId node_previous_left = cn.previousLeftId();
    NodeLocalId node_previous_right = cn.previousRightId();

    m_infos_view[node_previous_left].m_next_lid = node_next_left;
    m_infos_view[node_next_left].m_previous_lid = node_previous_left;

    m_infos_view[node_previous_right].m_next_lid = node_next_right;
    m_infos_view[node_next_right].m_previous_lid = node_previous_right;

    if (mesh_dim == 3) {
      NodeLocalId top_node_next_left = cn.topNextLeftId();
      NodeLocalId top_node_next_right = cn.topNextRightId();

      NodeLocalId top_node_previous_left = cn.topPreviousLeftId();
      NodeLocalId top_node_previous_right = cn.topPreviousRightId();

      m_infos_view[top_node_previous_left].m_next_lid = top_node_next_left;
      m_infos_view[top_node_next_left].m_previous_lid = top_node_previous_left;

      m_infos_view[top_node_previous_right].m_next_lid = top_node_next_right;
      m_infos_view[top_node_next_right].m_previous_lid = top_node_previous_right;
    }
  }

  UniqueArray<Int32> inner_cells_lid;
  UniqueArray<Int32> outer_cells_lid;
  cell_dm.innerCells().view().fillLocalIds(inner_cells_lid);
  cell_dm.outerCells().view().fillLocalIds(outer_cells_lid);

  UniqueArray<Int32> inner_lids;
  UniqueArray<Int32> outer_lids;
  // UniqueArray<Int32> inpatch_lids;
  // UniqueArray<Int32> overlap_lids;
  IItemFamily* family = all_nodes.itemFamily();
  ENUMERATE_ (Node, inode, all_nodes) {
    Int32 lid = inode.itemLocalId();
    Integer nb_inner_cells = 0;
    Integer nb_outer_cells = 0;
    for (Cell cell : inode->cells()) {
      if (inner_cells_lid.contains(cell.localId())) {
        nb_inner_cells++;
      }
      else if (outer_cells_lid.contains(cell.localId())) {
        nb_outer_cells++;
      }
    }
    if (nb_inner_cells + nb_outer_cells == inode->nbCell()) {
      inner_lids.add(lid);
    }
    else if (nb_outer_cells != 0) {
      outer_lids.add(lid);
    }

    // if (inode->hasFlags(ItemFlags::II_InPatch)) {
    //   inpatch_lids.add(lid);
    // }
    // if (inode->hasFlags(ItemFlags::II_Overlap)) {
    //   overlap_lids.add(lid);
    // }
  }
  int dir = (int)m_direction;
  String base_group_name = String("Direction") + dir;
  if (m_p->m_patch_index >= 0)
    base_group_name = base_group_name + String("AMRPatch") + m_p->m_patch_index;
  m_p->m_inner_all_items = family->createGroup(String("AllInner") + base_group_name, inner_lids, true);
  m_p->m_outer_all_items = family->createGroup(String("AllOuter") + base_group_name, outer_lids, true);
  // m_p->m_inpatch_all_items = family->createGroup(String("AllInPatch") + base_group_name, inpatch_lids, true);
  // m_p->m_overlap_all_items = family->createGroup(String("AllOverlap") + base_group_name, overlap_lids, true);
  m_p->m_inpatch_all_items = cell_dm.inPatchCells().nodeGroup();
  m_p->m_overlap_all_items = cell_dm.overlapCells().nodeGroup();
  m_p->m_all_items = all_nodes;

  _filterNodes();
  _computeNodeCellInfos();

  {
    UnstructuredMeshConnectivityView mesh_connectivity;
    mesh_connectivity.setMesh(m_p->m_cartesian_mesh->mesh());
    m_node_cell_view = mesh_connectivity.nodeCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Filters the front/back nodes to keep only the nodes of our patch.
 */
void NodeDirectionMng::
_filterNodes()
{
  // Set containing only the nodes of our patch
  std::set<NodeLocalId> nodes_set;
  ENUMERATE_NODE (inode, allNodes()) {
    nodes_set.insert(NodeLocalId(inode.itemLocalId()));
  }

  for (ItemDirectionInfo& idi : m_infos_view) {
    {
      Int32 next_lid = idi.m_next_lid;
      if (next_lid != NULL_ITEM_LOCAL_ID)
        if (nodes_set.find(NodeLocalId(next_lid)) == nodes_set.end())
          idi.m_next_lid = NodeLocalId{};
    }
    {
      Int32 prev_lid = idi.m_previous_lid;
      if (prev_lid != NULL_ITEM_LOCAL_ID)
        if (nodes_set.find(NodeLocalId(prev_lid)) == nodes_set.end())
          idi.m_previous_lid = NodeLocalId{};
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Brief calculation of node/cell connectivities by direction.
 */
void NodeDirectionMng::
_computeNodeCellInfos(const CellDirectionMng& cell_dm, const VariableCellReal3& cells_center)
{
  // TODO: only process the cells of our patch.
  IndexType indexes_ptr[8];
  ArrayView<IndexType> indexes(8, indexes_ptr);

  NodeDirectionMng& node_dm = *this;
  NodeGroup dm_all_nodes = node_dm.allNodes();
  eMeshDirection dir = m_direction;
  IMesh* mesh = m_p->m_cartesian_mesh->mesh();
  Integer mesh_dim = mesh->dimension();
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  if (mesh_dim != 2 && mesh_dim != 3)
    ARCANE_FATAL("Invalid mesh dimension '{0}'. Valid dimensions are 2 or 3", mesh_dim);

  // Set containing only the cells of our patch
  // This is used to filter to keep only these cells in the connectivity
  std::set<CellLocalId> inside_cells;
  ENUMERATE_CELL (icell, cell_dm.allCells()) {
    inside_cells.insert(CellLocalId(icell.itemLocalId()));
  }

  ENUMERATE_NODE (inode, dm_all_nodes) {
    Node node = *inode;
    Integer nb_cell = node.nbCell();
    Real3 node_pos = nodes_coord[node];
    indexes.fill(DirNode::NULL_CELL);
    for (Integer i = 0; i < nb_cell; ++i) {
      const IndexType bi = (IndexType)i;
      Cell cell = node.cell(i);
      if (inside_cells.find(CellLocalId(cell.localId())) == inside_cells.end())
        continue;

      Real3 center = cells_center[cell];
      Real3 wanted_cell_pos;
      Real3 wanted_node_pos;
      if (dir == MD_DirX) {
        wanted_cell_pos = center;
        wanted_node_pos = node_pos;
      }
      else if (dir == MD_DirY) {
        wanted_cell_pos = Real3(center.y, -center.x, center.z);
        wanted_node_pos = Real3(node_pos.y, -node_pos.x, node_pos.z);
      }
      else if (dir == MD_DirZ) {
        // TODO: to check for Y and Z
        wanted_cell_pos = Real3(center.z, -center.y, center.x);
        wanted_node_pos = Real3(node_pos.z, -node_pos.y, node_pos.x);
      }
      bool is_top = ((wanted_cell_pos.z > wanted_node_pos.z) && mesh_dim == 3);
      if (!is_top) {
        if (wanted_cell_pos.x > wanted_node_pos.x) {
          if (wanted_cell_pos.y > wanted_node_pos.y)
            indexes_ptr[CNP_NextLeft] = bi;
          else
            indexes_ptr[CNP_NextRight] = bi;
        }
        else {
          if (wanted_cell_pos.y > wanted_node_pos.y)
            indexes_ptr[CNP_PreviousLeft] = bi;
          else
            indexes_ptr[CNP_PreviousRight] = bi;
        }
      }
      else {
        if (wanted_cell_pos.x > wanted_node_pos.x) {
          if (wanted_cell_pos.y > wanted_node_pos.y)
            indexes_ptr[CNP_TopNextLeft] = bi;
          else
            indexes_ptr[CNP_TopNextRight] = bi;
        }
        else {
          if (wanted_cell_pos.y > wanted_node_pos.y)
            indexes_ptr[CNP_TopPreviousLeft] = bi;
          else
            indexes_ptr[CNP_TopPreviousRight] = bi;
        }
      }
    }
    m_infos_view[node.localId()].setCellIndexes(indexes_ptr);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_computeNodeCellInfos() const
{
  Ref<ICartesianMeshNumberingMngInternal> numbering = m_p->m_cartesian_mesh->_internalApi()->cartesianMeshNumberingMngInternal();

  IndexType indexes_ptr[8];
  ArrayView indexes(8, indexes_ptr);

  NodeGroup dm_all_nodes = this->allNodes();
  eMeshDirection dir = m_direction;
  IMesh* mesh = m_p->m_cartesian_mesh->mesh();
  Integer mesh_dim = mesh->dimension();

  if (mesh_dim == 2) {
    constexpr Integer nb_cells_max = 4;

    Int64 uids[nb_cells_max];
    ArrayView av_uids(nb_cells_max, uids);

    // DirX (Previous->X=0 / Next->X=1 / Right->Y=0 / Left->Y=1)
    // DirY (Previous->Y=0 / Next->Y=1 / Right->X=1 / Left->X=0)

    // The CartesianMeshNumberingMng always gives us the cells around the node in the same order:
    //
    // |2|3|
    //   .
    // |0|1|
    //
    // y
    // ^
    // |->x
    //
    // Read: the cell UID in the av_uids array at position 0 filled by
    // "numbering->cellUniqueIdsAroundNode(av_uids, node)" corresponds, in the X direction,
    // to the CNP_PreviousRight position.
    constexpr Int32 dir_x_pos_2d[nb_cells_max] = { CNP_PreviousRight, CNP_NextRight, CNP_PreviousLeft, CNP_NextLeft };
    constexpr Int32 dir_y_pos_2d[nb_cells_max] = { CNP_PreviousLeft, CNP_PreviousRight, CNP_NextLeft, CNP_NextRight };

    ENUMERATE_ (Node, inode, dm_all_nodes) {
      Node node = *inode;
      numbering->cellUniqueIdsAroundNode(node, av_uids);
      Integer nb_cell = node.nbCell();

      indexes.fill(DirNode::NULL_CELL);

      for (Integer i = 0; i < nb_cell; ++i) {
        Cell cell = node.cell(i);
        Integer pos = 0;
        for (; pos < nb_cells_max; ++pos) {
          if (cell.uniqueId() == av_uids[pos])
            break;
        }
        if (pos == nb_cells_max)
          continue;

        const IndexType bi = (IndexType)i;
        if (dir == MD_DirX) {
          indexes[dir_x_pos_2d[pos]] = bi;
        }
        else if (dir == MD_DirY) {
          indexes[dir_y_pos_2d[pos]] = bi;
        }
      }
      m_infos_view[node.localId()].setCellIndexes(indexes_ptr);
    }
  }
  else if (mesh_dim == 3) {
    constexpr Integer nb_cells_max = 8;

    Int64 uids[nb_cells_max];
    ArrayView av_uids(nb_cells_max, uids);

    // DirX (Top->Z=1 / Previous->X=0 / Next->X=1 / Right->Y=0 / Left->Y=1)
    // DirY (Top->Z=1 / Previous->Y=0 / Next->Y=1 / Right->X=1 / Left->X=0)
    // DirZ (Top->Y=1 / Previous->Z=0 / Next->Z=1 / Right->X=1 / Left->X=0)

    // The CartesianMeshNumberingMng always gives us the cells around the node in the same order:
    //
    // z = 0 | z = 1
    // |2|3| | |6|7|
    //   .   |   .
    // |0|1| | |4|5|
    //
    // y
    // ^
    // |->x
    //
    // Read: the cell UID in the av_uids array at position 2 filled by
    // "numbering->cellUniqueIdsAroundNode(av_uids, node)" corresponds, in the Z direction,
    // to the CNP_TopPreviousLeft position.
    constexpr Int32 dir_x_pos_3d[nb_cells_max] = { CNP_PreviousRight, CNP_NextRight, CNP_PreviousLeft, CNP_NextLeft, CNP_TopPreviousRight, CNP_TopNextRight, CNP_TopPreviousLeft, CNP_TopNextLeft };
    constexpr Int32 dir_y_pos_3d[nb_cells_max] = { CNP_PreviousLeft, CNP_PreviousRight, CNP_NextLeft, CNP_NextRight, CNP_TopPreviousLeft, CNP_TopPreviousRight, CNP_TopNextLeft, CNP_TopNextRight };
    constexpr Int32 dir_z_pos_3d[nb_cells_max] = { CNP_PreviousLeft, CNP_PreviousRight, CNP_TopPreviousLeft, CNP_TopPreviousRight, CNP_NextLeft, CNP_NextRight, CNP_TopNextLeft, CNP_TopNextRight };

    ENUMERATE_ (Node, inode, dm_all_nodes) {
      Node node = *inode;
      numbering->cellUniqueIdsAroundNode(node, av_uids);
      Integer nb_cell = node.nbCell();

      indexes.fill(DirNode::NULL_CELL);

      for (Integer i = 0; i < nb_cell; ++i) {
        Cell cell = node.cell(i);
        Integer pos = 0;
        for (; pos < nb_cells_max; ++pos) {
          if (cell.uniqueId() == av_uids[pos])
            break;
        }
        if (pos == nb_cells_max)
          continue;

        const IndexType bi = (IndexType)i;

        if (dir == MD_DirX) {
          indexes[dir_x_pos_3d[pos]] = bi;
        }
        else if (dir == MD_DirY) {
          indexes[dir_y_pos_3d[pos]] = bi;
        }
        else if (dir == MD_DirZ) {
          indexes[dir_z_pos_3d[pos]] = bi;
        }

        m_infos_view[node.localId()].setCellIndexes(indexes_ptr);
      }
    }
  }
  else {
    ARCANE_FATAL("Invalid mesh dimension '{0}'. Valid dimensions are 2 or 3", mesh_dim);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup NodeDirectionMng::
allNodes() const
{
  return m_p->m_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup NodeDirectionMng::
overlapNodes() const
{
  return m_p->m_overlap_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup NodeDirectionMng::
inPatchNodes() const
{
  return m_p->m_inpatch_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup NodeDirectionMng::
innerNodes() const
{
  return m_p->m_inner_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup NodeDirectionMng::
outerNodes() const
{
  return m_p->m_outer_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
