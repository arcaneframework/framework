// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeDirectionMng.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Infos sur les mailles d'une direction X Y ou Z d'un maillage structuré.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/NodeDirectionMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"
#include "arcane/VariableTypes.h"
#include "arcane/UnstructuredMeshConnectivity.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"

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

  Impl() : m_infos(platform::getDefaultDataAllocator()){}

 public:

  NodeGroup m_inner_all_items;
  NodeGroup m_outer_all_items;
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
_internalInit(ICartesianMesh* cm,eMeshDirection dir,Integer patch_index)
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
_internalComputeInfos(const CellDirectionMng& cell_dm,const NodeGroup& all_nodes,
                      const VariableCellReal3& cells_center)
{
  Node null_node;
  m_infos_view.fill(NodeDirectionMng::ItemDirectionInfo());

  Integer mesh_dim = m_p->m_cartesian_mesh->mesh()->dimension();
  //TODO: ne garder que les noeuds de notre patch

  // Calcul les infos de direction pour les noeuds
  ENUMERATE_CELL(icell,cell_dm.allCells()){
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

    if (mesh_dim==3){
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
  cell_dm.innerCells().view().fillLocalIds(inner_cells_lid);

  UniqueArray<Int32> inner_lids;
  UniqueArray<Int32> outer_lids;
  IItemFamily* family = all_nodes.itemFamily();
  ENUMERATE_ (Node, inode, all_nodes) {
    Int32 lid = inode.itemLocalId();
    Int32 i1 = m_infos_view[lid].m_next_lid;
    Int32 i2 = m_infos_view[lid].m_previous_lid;
    if (i1 == NULL_ITEM_LOCAL_ID || i2 == NULL_ITEM_LOCAL_ID) {
      outer_lids.add(lid);
    }
    else {
      // Si au moins une des mailles est interne, alors le noeud est interne.
      // TODO : Gérer les noeuds communs peut-être une autre méthode plus performante.
      bool is_outer = true;
      for (Cell cell : inode->cells()) {
        if (inner_cells_lid.contains(cell.localId())) {
          inner_lids.add(lid);
          is_outer = false;
          break;
        }
      }
      if (is_outer) {
        outer_lids.add(lid);
      }
    }
  }
  int dir = (int)m_direction;
  String base_group_name = String("Direction")+dir;
  if (m_p->m_patch_index>=0)
    base_group_name = base_group_name + String("AMRPatch")+m_p->m_patch_index;
  m_p->m_inner_all_items = family->createGroup(String("AllInner")+base_group_name,inner_lids,true);
  m_p->m_outer_all_items = family->createGroup(String("AllOuter")+base_group_name,outer_lids,true);
  m_p->m_all_items = all_nodes;

  _filterNodes();
  _computeNodeCellInfos(cell_dm,cells_center);

  {
    UnstructuredMeshConnectivityView mesh_connectivity;
    mesh_connectivity.setMesh(m_p->m_cartesian_mesh->mesh());
    m_node_cell_view = mesh_connectivity.nodeCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Filtre les noeuds devant/derrière pour ne garder que les
 * noeuds de notre patch.
 */
void NodeDirectionMng::
_filterNodes()
{
  // Ensemble contenant uniquement les noeuds de notre patch
  std::set<NodeLocalId> nodes_set;
  ENUMERATE_NODE(inode,allNodes()){
    nodes_set.insert(NodeLocalId(inode.itemLocalId()));
  }

  for( ItemDirectionInfo& idi : m_infos_view ){
    {
      Int32 next_lid = idi.m_next_lid;
      if (next_lid!=NULL_ITEM_LOCAL_ID)
        if (nodes_set.find(NodeLocalId(next_lid))==nodes_set.end())
          idi.m_next_lid = NodeLocalId{};
    }
    {
      Int32 prev_lid = idi.m_previous_lid;
      if (prev_lid!=NULL_ITEM_LOCAL_ID)
        if (nodes_set.find(NodeLocalId(prev_lid))==nodes_set.end())
          idi.m_previous_lid = NodeLocalId{};
    }
  }
}
 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des connectivités noeuds/mailles par direction.
 */
void NodeDirectionMng::
_computeNodeCellInfos(const CellDirectionMng& cell_dm,const VariableCellReal3& cells_center)
{
  // TODO: ne traiter que les mailles de notre patch.
  IndexType indexes_ptr[8];
  ArrayView<IndexType> indexes(8,indexes_ptr);

  NodeDirectionMng& node_dm = *this;
  NodeGroup dm_all_nodes = node_dm.allNodes();
  eMeshDirection dir = m_direction;
  IMesh* mesh = m_p->m_cartesian_mesh->mesh();
  Integer mesh_dim = mesh->dimension();
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  if (mesh_dim!=2 && mesh_dim!=3)
    ARCANE_FATAL("Invalid mesh dimension '{0}'. Valid dimensions are 2 or 3",mesh_dim);

  // Ensemble contenant uniquement les mailles de notre patch
  // Cela sert à filtrer pour ne garder que ces mailles là dans la connectivité
  std::set<CellLocalId> inside_cells;
  ENUMERATE_CELL(icell,cell_dm.allCells()){
    inside_cells.insert(CellLocalId(icell.itemLocalId()));
  }
  
  ENUMERATE_NODE(inode,dm_all_nodes){
    Node node = *inode;
    Integer nb_cell = node.nbCell();
    Real3 node_pos = nodes_coord[node];
    indexes.fill(DirNode::NULL_CELL);
    for( Integer i=0; i<nb_cell; ++i ){
      const IndexType bi = (IndexType)i;
      Cell cell = node.cell(i);
      if (inside_cells.find(CellLocalId(cell.localId()))==inside_cells.end())
        continue;

      Real3 center = cells_center[cell];
      Real3 wanted_cell_pos;
      Real3 wanted_node_pos;
      if (dir==MD_DirX){
        wanted_cell_pos = center;
        wanted_node_pos = node_pos;
      } else if (dir==MD_DirY){
        wanted_cell_pos = Real3(center.y, -center.x, center.z);
        wanted_node_pos = Real3(node_pos.y, -node_pos.x, node_pos.z);
      } else if (dir==MD_DirZ){
        // TODO: à vérifier pour Y et Z
        wanted_cell_pos = Real3(center.z, -center.y, center.x);
        wanted_node_pos = Real3(node_pos.z, -node_pos.y, node_pos.x);
      }
      bool is_top = ((wanted_cell_pos.z > wanted_node_pos.z) && mesh_dim==3);
      if (!is_top){
        if (wanted_cell_pos.x > wanted_node_pos.x ){
          if (wanted_cell_pos.y > wanted_node_pos.y )
            indexes_ptr[CNP_NextLeft] = bi;
          else
            indexes_ptr[CNP_NextRight] = bi;
        }
        else{
          if (wanted_cell_pos.y > wanted_node_pos.y )
            indexes_ptr[CNP_PreviousLeft] = bi;
          else
            indexes_ptr[CNP_PreviousRight] = bi;
        }
      }
      else{
        if (wanted_cell_pos.x > wanted_node_pos.x ){
          if (wanted_cell_pos.y > wanted_node_pos.y )
            indexes_ptr[CNP_TopNextLeft] = bi;
          else
            indexes_ptr[CNP_TopNextRight] = bi;
        }
        else{
          if (wanted_cell_pos.y > wanted_node_pos.y )
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

NodeGroup NodeDirectionMng::
allNodes() const
{
  return m_p->m_all_items;
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
