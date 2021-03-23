// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfo.cc                                           (C) 2000-2020 */
/*                                                                           */
/* Informations communes à plusieurs entités.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemSharedInfo.h"

#include "arcane/utils/Iostream.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo ItemSharedInfo::nullItemSharedInfo;

bool ItemSharedInfo::m_is_amr_activated = false;

// Suppose NULL_ITEM_UNIQUE_ID == (-1) et NULL_ITEM_LOCAL_ID == (-1)
// Cree un pseudo-tableau qui pourra etre indexé avec NULL_ITEM_LOCAL_ID
// pour la maille nulle.
static Int64 null_int64_buf[2] = { NULL_ITEM_UNIQUE_ID, NULL_ITEM_UNIQUE_ID };
static Int64ArrayView null_unique_ids(1,null_int64_buf + 1);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo()
: m_connectivity(&ItemInternalConnectivityList::nullInstance)
, m_unique_ids(&null_unique_ids)
{
  _init(IK_Unknown);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,
               MeshItemInternalList* items,ItemInternalConnectivityList* connectivity,
               Int64ArrayView* unique_ids)
: m_nb_node(item_type->nbLocalNode())
, m_nb_edge(item_type->nbLocalEdge())
, m_nb_face(item_type->nbLocalFace())
, m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(unique_ids)
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_type_id(item_type->typeId())
{
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
               Int32 nb_edge,Int32 nb_face,Int32 nb_cell)
: m_nb_node(item_type->nbLocalNode())
, m_nb_edge(nb_edge)
, m_nb_face(nb_face)
, m_nb_cell(nb_cell)
, m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(unique_ids)
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_edge_allocated(m_nb_edge)
, m_face_allocated(m_nb_face)
, m_cell_allocated(m_nb_cell)
, m_hParent_allocated(m_nb_hParent)
, m_hChild_allocated(m_nb_hChildren)
, m_type_id(item_type->typeId())
{
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
               Int32 nb_edge,Int32 nb_face,Int32 nb_cell,
               Int32 edge_allocated,Int32 face_allocated,Int32 cell_allocated)
: m_nb_node(item_type->nbLocalNode())
, m_nb_edge(nb_edge)
, m_nb_face(nb_face)
, m_nb_cell(nb_cell)
, m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(unique_ids)
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_edge_allocated(edge_allocated)
, m_face_allocated(face_allocated)
, m_cell_allocated(cell_allocated)
, m_type_id(item_type->typeId())
{
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
               Int32 nb_edge,Int32 nb_face,Int32 nb_cell,
               Int32 nb_hParent,Int32 nb_hChildren,
               Int32 edge_allocated,Int32 face_allocated,Int32 cell_allocated,
               Int32 hParent_allocated,Int32 hChild_allocated)
:  m_nb_node(item_type->nbLocalNode())
, m_nb_edge(nb_edge)
, m_nb_face(nb_face)
, m_nb_cell(nb_cell)
, m_nb_hParent(nb_hParent)
, m_nb_hChildren(nb_hChildren)
, m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(unique_ids)
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_edge_allocated(edge_allocated)
, m_face_allocated(face_allocated)
, m_cell_allocated(cell_allocated)
, m_hParent_allocated(hParent_allocated)
, m_hChild_allocated(hChild_allocated)
, m_type_id(item_type->typeId())
{
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
               Int32ConstArrayView buffer)
: m_nb_node(item_type->nbLocalNode())
, m_nb_edge(buffer[1])
, m_nb_face(buffer[2])
, m_nb_cell(buffer[3])
, m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(unique_ids)
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_edge_allocated(buffer[4])
, m_face_allocated(buffer[5])
, m_cell_allocated(buffer[6])
, m_type_id(item_type->typeId())
, m_index(buffer[7])
, m_nb_reference(buffer[8])
{
  //! AMR
  if (buffer.size()>=serializeAMRSize()){
    m_nb_hParent = buffer[9];
    m_nb_hChildren = buffer[10];
    m_hParent_allocated = buffer[11];
    m_hChild_allocated = buffer[12];
  }
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfo::
serializeSize()
{
	//! AMR
	if(ItemSharedInfo::m_is_amr_activated)
		return serializeAMRSize();
  return serializeNoAMRSize();
}

Integer ItemSharedInfo::
serializeAMRSize()
{
  return 13;
}

Integer ItemSharedInfo::
serializeNoAMRSize()
{
  return 9;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
serializeWrite(Int32ArrayView buffer)
{
  buffer[0] = m_type_id; // Doit toujours être le premier

  buffer[1] = m_nb_edge;
  buffer[2] = m_nb_face;
  buffer[3] = m_nb_cell;

  buffer[4] = m_edge_allocated;
  buffer[5] = m_face_allocated;
  buffer[6] = m_cell_allocated;

  buffer[7] = m_index;
  buffer[8] = m_nb_reference;

  //! AMR
  if (buffer.size()>=serializeAMRSize()){
	  buffer[9] = m_nb_hParent;
	  buffer[10] = m_nb_hChildren;
	  buffer[11] = m_hParent_allocated;
	  buffer[12] = m_hChild_allocated;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
print(std::ostream& o) const
{
  o << " This: " << this
    << " Infos: " << m_infos
    << " FirstNode: " << m_first_node
    << " NbNode: " << m_nb_node
    << " FirstEdge: " << m_first_edge
    << " NbEdge: " << m_nb_edge
    << " FirstFace: " << m_first_face
    << " NbFace: " << m_nb_face
    << " FirstCell: " << m_first_cell
    << " NbCell: " << m_nb_cell
    << " FirstParent: " << m_first_parent
    << " NbParent; " << m_nb_parent
    //! AMR
    << " FirsthParent: " << m_first_hParent
    << " NbhParent: " << m_nb_hParent
    << " FirsthChild: " << m_first_hChild
    << " NbhChildren: " << m_nb_hChildren
    << " hParentAlloc: " << m_hParent_allocated
    << " hChildAlloc: " << m_hChild_allocated
    // OFF AMR
    << " Items: " << m_items
    << " Connectivity: " << m_connectivity
    << " Family: " << m_item_family->fullName()
    << " TypeInfo: " << m_item_type
    << " NeededMemory: " << m_needed_memory
    << " MiniumMemory: " << m_minimum_needed_memory
    << " EdgeAlloc: " << m_edge_allocated
    << " FaceAlloc: " << m_face_allocated
    << " CellAlloc: " << m_cell_allocated
    << " TypeId: " << m_type_id
    << " Index: " << m_index
    << " NbReference: " << m_nb_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
_init(eItemKind ik)
{
  if (ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell){
    IItemFamily* base_family = m_item_family;
    m_nb_parent = base_family->parentFamilyDepth();
    ARCANE_ASSERT((m_nb_parent<=1),("More than one parent level: not implemented"));
    ItemSharedInfo::m_is_amr_activated = m_items->mesh->isAmrActivated();
  }

  m_first_node = FIRST_NODE_INDEX;
  m_first_edge = m_first_node + m_nb_node;
  m_first_face = m_first_edge + m_edge_allocated;
  m_first_cell = m_first_face + m_face_allocated;
  m_first_parent = m_first_cell + m_cell_allocated;

  //! AMR
  if(ItemSharedInfo::m_is_amr_activated){
    m_first_hParent = m_first_parent + m_nb_parent;
    m_first_hChild = m_first_hParent + m_hParent_allocated;
    m_needed_memory = m_first_hChild + m_hChild_allocated;
  }
  else {
    m_needed_memory = m_first_parent + m_nb_parent;
  }
  m_minimum_needed_memory =
    COMMON_BASE_MEMORY
    + m_nb_node
    + m_nb_edge
    + m_nb_face
    + m_nb_cell
    + m_nb_parent;

  //! AMR
  if(ItemSharedInfo::m_is_amr_activated)
   	m_minimum_needed_memory +=	m_nb_hParent + m_nb_hChildren;

  //! Indique si on active ou pas les anciennes connectivités
  if (m_item_family){
    InternalConnectivityPolicy icp = m_item_family->mesh()->_connectivityPolicy();
    m_has_legacy_connectivity = InternalConnectivityInfo::hasLegacyConnectivity(icp);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalArrayView ItemSharedInfo::
_parents(Integer index) const
{
  ARCANE_ASSERT((index==0),("Only one parent access implemented"));
#ifndef NO_USER_WARNING
#warning "(HP) TODO: à optimiser comme m_dual_items"
#endif /* NO_USER_WARNING */
  // En pointant vers le bon champ du MeshItemInternalList dans le maillage parent
  return m_items->mesh->itemFamily(m_item_kind)->parentFamily()->itemsInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

