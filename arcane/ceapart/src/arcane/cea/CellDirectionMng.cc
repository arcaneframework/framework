// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellDirectionMng.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Infos sur les mailles d'une direction X Y ou Z d'un maillage structuré.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"

#include "arcane/cea/CellDirectionMng.h"
#include "arcane/cea/ICartesianMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellDirectionMng::Impl
{
 public:
  CellGroup m_inner_all_items;
  CellGroup m_outer_all_items;
  CellGroup m_all_items;
  ICartesianMesh* m_cartesian_mesh = nullptr;
  Integer m_patch_index = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellDirectionMng::
CellDirectionMng()
: m_direction(MD_DirInvalid)
, m_p(nullptr)
, m_next_face_index(-1)
, m_previous_face_index(-1)
, m_sub_domain_offset(-1)
, m_own_nb_cell(-1)
, m_global_nb_cell(-1)
, m_own_cell_offset(-1)
{
  for( Integer i=0; i<MAX_NB_NODE; ++i )
    m_nodes_indirection[i] = (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellDirectionMng::
CellDirectionMng(const CellDirectionMng& rhs)
: m_infos(rhs.m_infos)
, m_direction(rhs.m_direction)
, m_p(rhs.m_p)
, m_items(rhs.m_p->m_all_items.itemFamily()->itemsInternal())
, m_next_face_index(rhs.m_next_face_index)
, m_previous_face_index(rhs.m_previous_face_index)
, m_sub_domain_offset(rhs.m_sub_domain_offset)
, m_own_nb_cell(rhs.m_own_nb_cell)
, m_global_nb_cell(rhs.m_global_nb_cell)
, m_own_cell_offset(rhs.m_own_cell_offset)
{
  for( Integer i=0; i<MAX_NB_NODE; ++i )
    m_nodes_indirection[i] = rhs.m_nodes_indirection[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellDirectionMng::
~CellDirectionMng()
{
  // Ne pas détruire le m_p.
  // Le gestionnnaire le fera via destroy()
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
_internalComputeInnerAndOuterItems(const ItemGroup& items)
{
  Int32UniqueArray inner_lids;
  Int32UniqueArray outer_lids;
  IItemFamily* family = items.itemFamily();
  ENUMERATE_ITEM(iitem,items){
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
  m_p->m_all_items = items;
  m_items = family->itemsInternal();
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
setNodesIndirection(Int32ConstArrayView nodes_indirection)
{
  for( Integer i=0; i<MAX_NB_NODE; ++i )
    m_nodes_indirection[i] = nodes_indirection[i];

  ITraceMng* tm = m_p->m_cartesian_mesh->traceMng();

  tm->info() << "Set computed indirection dir=" << (int)m_direction;
  for( Integer i=0; i<MAX_NB_NODE; ++i ){
    tm->info() << "Indirection i=" << i << " v=" << m_nodes_indirection[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
