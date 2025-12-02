// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellDirectionMng.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Infos sur les mailles d'une direction X Y ou Z d'un maillage structuré.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"

#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellDirectionMng::Impl
{
 public:

  Impl() : m_infos(platform::getDefaultDataAllocator()){}

 public:

  CellGroup m_inner_all_items;
  CellGroup m_outer_all_items;
  CellGroup m_inpatch_all_items;
  CellGroup m_overall_all_items;
  CellGroup m_all_items;
  ICartesianMesh* m_cartesian_mesh = nullptr;
  Integer m_patch_index = -1;
  UniqueArray<ItemDirectionInfo> m_infos;
  Int32 m_sub_domain_offset = -1;
  Int32 m_own_nb_cell = -1;
  Int64 m_global_nb_cell = -1;
  Int64 m_own_cell_offset = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellDirectionMng::
CellDirectionMng()
: m_direction(MD_DirInvalid)
, m_next_face_index(-1)
, m_previous_face_index(-1)
{
  for( Integer i=0; i<MAX_NB_NODE; ++i )
    m_nodes_indirection[i] = (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellDirectionMng::
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

void CellDirectionMng::
_internalDestroy()
{
  delete m_p;
  m_p = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellDirectionMng::
_internalResizeInfos(Int32 new_size)
{
  m_p->m_infos.resize(new_size);
  m_infos_view = m_p->m_infos.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellDirectionMng::
_internalComputeInnerAndOuterItems(const ItemGroup& items)
{
  Int32UniqueArray inner_lids;
  Int32UniqueArray outer_lids;
  IItemFamily* family = items.itemFamily();
  ENUMERATE_ITEM(iitem,items){
    Int32 lid = iitem.itemLocalId();
    Int32 i1 = m_infos_view[lid].m_next_lid;
    Int32 i2 = m_infos_view[lid].m_previous_lid;
    if (i1 == NULL_ITEM_LOCAL_ID || i2 == NULL_ITEM_LOCAL_ID)
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
  m_p->m_all_items = items;
  m_cells = CellInfoListView(family);

  UnstructuredMeshConnectivityView mesh_connectivity;
  mesh_connectivity.setMesh(m_p->m_cartesian_mesh->mesh());
  m_cell_node_view = mesh_connectivity.cellNode();
  m_cell_face_view = mesh_connectivity.cellFace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellDirectionMng::
_internalComputeCellGroups(const CellGroup& all_cells, const CellGroup& in_patch_cells, const CellGroup& overall_cells)
{
  m_p->m_inpatch_all_items = in_patch_cells;
  m_p->m_overall_all_items = overall_cells;
  m_p->m_all_items = all_cells;

  UniqueArray<Int32> overall_lid;
  overall_cells.view().fillLocalIds(overall_lid);

  UniqueArray<Int32> inner_lids;
  UniqueArray<Int32> outer_lids;

  ENUMERATE_ (Cell, icell, in_patch_cells) {
    Int32 lid = icell.itemLocalId();
    Int32 i1 = m_infos_view[lid].m_next_lid;
    Int32 i2 = m_infos_view[lid].m_previous_lid;
    if (i1 == NULL_ITEM_LOCAL_ID || i2 == NULL_ITEM_LOCAL_ID || overall_lid.contains(i1) || overall_lid.contains(i2))
      outer_lids.add(lid);
    else
      inner_lids.add(lid);
  }
  int dir = (int)m_direction;
  IItemFamily* family = all_cells.itemFamily();
  String base_group_name = String("Direction") + dir;
  if (m_p->m_patch_index >= 0)
    base_group_name = base_group_name + String("AMRPatch") + m_p->m_patch_index;
  m_p->m_inner_all_items = family->createGroup(String("AllInner") + base_group_name, inner_lids, true);
  m_p->m_outer_all_items = family->createGroup(String("AllOuter") + base_group_name, outer_lids, true);
  m_cells = CellInfoListView(family);

  UnstructuredMeshConnectivityView mesh_connectivity;
  mesh_connectivity.setMesh(m_p->m_cartesian_mesh->mesh());
  m_cell_node_view = mesh_connectivity.cellNode();
  m_cell_face_view = mesh_connectivity.cellFace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CellDirectionMng::
allCells() const
{
  return m_p->m_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CellDirectionMng::
overallCells() const
{
  return m_p->m_overall_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CellDirectionMng::
inPatchCells() const
{
  return m_p->m_inpatch_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CellDirectionMng::
innerCells() const
{
  return m_p->m_inner_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CellDirectionMng::
outerCells() const
{
  return m_p->m_outer_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellDirectionMng::
setNodesIndirection(ConstArrayView<Int8> nodes_indirection)
{
  for( Integer i=0; i<MAX_NB_NODE; ++i )
    m_nodes_indirection[i] = nodes_indirection[i];

  ITraceMng* tm = m_p->m_cartesian_mesh->traceMng();

  tm->info(4) << "Set computed indirection dir=" << (int)m_direction;
  for( Integer i=0; i<MAX_NB_NODE; ++i ){
    tm->info(5) << "Indirection i=" << i << " v=" << (int)m_nodes_indirection[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CellDirectionMng::
globalNbCell() const
{
  return (m_p) ? m_p->m_global_nb_cell : -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CellDirectionMng::
ownNbCell() const
{
  return (m_p) ? m_p->m_own_nb_cell : -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CellDirectionMng::
subDomainOffset() const
{
  return (m_p) ? m_p->m_sub_domain_offset : -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CellDirectionMng::
ownCellOffset() const
{
  return (m_p) ? m_p->m_own_cell_offset : -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellDirectionMng::
_internalSetOffsetAndNbCellInfos(Int64 global_nb_cell, Int32 own_nb_cell,
                                 Int32 sub_domain_offset, Int64 own_cell_offset)
{
  m_p->m_global_nb_cell = global_nb_cell;
  m_p->m_own_nb_cell = own_nb_cell;
  m_p->m_sub_domain_offset = sub_domain_offset;
  m_p->m_own_cell_offset = own_cell_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
