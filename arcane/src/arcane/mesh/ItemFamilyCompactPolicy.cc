// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyCompactPolicy.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Politique de compactage des entités.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"
#include "arcane/IMeshCompacter.h"
#include "arcane/ItemFamilyCompactInfos.h"

#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamilyCompactPolicy::
ItemFamilyCompactPolicy(ItemFamily* family)
: TraceAccessor(family->traceMng())
, m_family(family)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
beginCompact(ItemFamilyCompactInfos& compact_infos)
{
  m_family->beginCompactItems(compact_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos)
{
  m_family->compactVariablesAndGroups(compact_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
endCompact(ItemFamilyCompactInfos& compact_infos)
{
  m_family->finishCompactItems(compact_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
compactConnectivityData()
{
  m_family->compactReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemFamilyCompactPolicy::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardItemFamilyCompactPolicy::
StandardItemFamilyCompactPolicy(ItemFamily* family)
: ItemFamilyCompactPolicy(family)
{
  IMesh* mesh = family->mesh();

  m_node_family = mesh->nodeFamily();
  m_edge_family = mesh->edgeFamily();
  m_face_family = mesh->faceFamily();
  m_cell_family = mesh->cellFamily();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardItemFamilyCompactPolicy::
updateInternalReferences(IMeshCompacter* compacter)
{
  // On ne met à jour les références que si on a les anciennes connectivités.
  InternalConnectivityPolicy icp = _family()->mesh()->_connectivityPolicy();
  bool has_legacy_connectivity = InternalConnectivityInfo::hasLegacyConnectivity(icp);
  if (!has_legacy_connectivity)
    return;

  Int32ConstArrayView nodes_old_to_new_lid;
  Int32ConstArrayView edges_old_to_new_lid;
  Int32ConstArrayView faces_old_to_new_lid;
  Int32ConstArrayView cells_old_to_new_lid;

  const ItemFamilyCompactInfos* node_infos = compacter->findCompactInfos(m_node_family);
  const ItemFamilyCompactInfos* edge_infos = compacter->findCompactInfos(m_edge_family);
  const ItemFamilyCompactInfos* face_infos = compacter->findCompactInfos(m_face_family);
  const ItemFamilyCompactInfos* cell_infos = compacter->findCompactInfos(m_cell_family);

  if (node_infos)
    nodes_old_to_new_lid = node_infos->oldToNewLocalIds();
  if (edge_infos)
    edges_old_to_new_lid = edge_infos->oldToNewLocalIds();
  if (face_infos)
    faces_old_to_new_lid = face_infos->oldToNewLocalIds();
  if (cell_infos)
    cells_old_to_new_lid = cell_infos->oldToNewLocalIds();

  ENUMERATE_ITEM(iitem,_family()->allItems()){
    ItemInternal* item = (*iitem).internal();

    if (node_infos)
      _changeItem(item->_nodesPtr(),item->nbNode(),nodes_old_to_new_lid);
    if (edge_infos)
      _changeItem(item->_edgesPtr(),item->nbEdge(),edges_old_to_new_lid);
    if (face_infos)
      _changeItem(item->_facesPtr(),item->nbFace(),faces_old_to_new_lid);
    if (cell_infos){
      _changeItem(item->_cellsPtr(),item->nbCell(),cells_old_to_new_lid);
      //! AMR
      _changeItem(item->_hParentPtr(),item->nbHParent(),cells_old_to_new_lid);
      _changeItem(item->_hChildPtr(),item->nbHChildren(),cells_old_to_new_lid);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
