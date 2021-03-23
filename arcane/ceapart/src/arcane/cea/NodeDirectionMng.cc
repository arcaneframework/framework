// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeDirectionMng.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Infos sur les mailles d'une direction X Y ou Z d'un maillage structuré.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cea/NodeDirectionMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"

#include "arcane/cea/ICartesianMesh.h"
#include "arcane/cea/CellDirectionMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeDirectionMng::Impl
{
 public:
  NodeGroup m_inner_all_items;
  NodeGroup m_outer_all_items;
  NodeGroup m_all_items;
  ICartesianMesh* m_cartesian_mesh = nullptr;
  Integer m_patch_index = -1;
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

NodeDirectionMng::
NodeDirectionMng(const NodeDirectionMng& rhs)
: m_infos(rhs.m_infos)
, m_direction(rhs.m_direction)
, m_p(rhs.m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeDirectionMng::
~NodeDirectionMng()
{
  // Ne pas détruire le m_p.
  // Le gestionnnaire le fera via destroy()
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeDirectionMng::
_internalInit(ICartesianMesh* cm,eMeshDirection dir,Integer patch_index)
{
  if (m_p)
    ARCANE_FATAL("Initialisation already done");
  m_p = new Impl();
  m_direction = dir;
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
_internalComputeInfos(const CellDirectionMng& cell_dm,const NodeGroup& all_nodes)
{
  Node null_node;
  m_infos.fill(NodeDirectionMng::ItemDirectionInfo(null_node.internal(),null_node.internal()));

  Integer mesh_dim = m_p->m_cartesian_mesh->mesh()->dimension();

  // Calcul les infos de direction pour les noeuds
  ENUMERATE_CELL(icell,cell_dm.allCells()){
    Cell cell = *icell;
    DirCellNode cn(cell_dm.cellNode(cell));

    Node node_next_left = cn.nextLeft();
    Node node_next_right = cn.nextRight();

    Node node_previous_left = cn.previousLeft();
    Node node_previous_right = cn.previousRight();

    m_infos[node_previous_left.localId()].m_next_item = node_next_left.internal();
    m_infos[node_next_left.localId()].m_previous_item = node_previous_left.internal();

    m_infos[node_previous_right.localId()].m_next_item = node_next_right.internal();
    m_infos[node_next_right.localId()].m_previous_item = node_previous_right.internal();

    if (mesh_dim==3){
      Node top_node_next_left = cn.topNextLeft();
      Node top_node_next_right = cn.topNextRight();

      Node top_node_previous_left = cn.topPreviousLeft();
      Node top_node_previous_right = cn.topPreviousRight();

      m_infos[top_node_previous_left.localId()].m_next_item = top_node_next_left.internal();
      m_infos[top_node_next_left.localId()].m_previous_item = top_node_previous_left.internal();

      m_infos[top_node_previous_right.localId()].m_next_item = top_node_next_right.internal();
      m_infos[top_node_next_right.localId()].m_previous_item = top_node_previous_right.internal();
    }
  }

  Int32UniqueArray inner_lids;
  Int32UniqueArray outer_lids;
  IItemFamily* family = all_nodes.itemFamily();
  ENUMERATE_ITEM(iitem,all_nodes){
    Int32 lid = iitem.itemLocalId();
    ItemInternal* i1 = m_infos[lid].m_next_item;
    ItemInternal* i2 = m_infos[lid].m_previous_item;
    if (i1->null() || i2->null())
      outer_lids.add(lid);
    else
      inner_lids.add(lid);
  }
  int dir = (int)m_direction;
  String base_group_name = String("Direction")+dir;
  if (m_p->m_patch_index>=0)
    base_group_name = base_group_name + String("AMRPatch")+m_p->m_patch_index;
  m_p->m_inner_all_items = family->createGroup(String("AllInner")+base_group_name,inner_lids,true);
  m_p->m_outer_all_items = family->createGroup(String("AllOuter")+base_group_name,outer_lids,true);
  m_p->m_all_items = all_nodes;
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
