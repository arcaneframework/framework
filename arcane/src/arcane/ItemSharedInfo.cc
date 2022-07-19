// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfo.cc                                           (C) 2000-2022 */
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
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,ItemVariableViews* variable_views)
: m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(&(variable_views->m_unique_ids_view))
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_type_id(item_type->typeId())
{
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Constructeur de désérialisation
ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,ItemVariableViews* variable_views,
               Int32ConstArrayView buffer)
: m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_unique_ids(&(variable_views->m_unique_ids_view))
, m_item_type(item_type)
, m_item_kind(family->itemKind())
, m_type_id(item_type->typeId())
{
  // La taille du buffer dépend des versions de Arcane.
  // Avant la 3.2 (Octobre 2021), la taille du buffer est 9 (non AMR) ou 13 (AMR)
  // Entre la 3.2 et la 3.6 (Mai 2022), la taille vaut toujours 13
  // A partir de la 3.6, la taille vaut 6.
  //
  // On ne cherche pas à faire de reprise avec des versions
  // de Arcane antérieures à 3.2 donc on peut supposer que la taille
  // du buffer vaut 13. Ces versions utilisent la nouvelle connectivité
  // et donc le nombre des éléments est toujours 0 (ainsi que les *allocated)
  // sauf pour m_nb_node.
  //
  // A partir de la 3.6, le nombre de noeuds n'est plus utilisé non
  // plus et vaut toujours 0. On pourra donc pour les versions de fin
  // 2022 supprimer ces champs de ItemSharedInfo.
  Int32 buf_size = buffer.size();
  if (buf_size>=9){
    m_nb_node = item_type->nbLocalNode();
    m_nb_edge = buffer[1];
    m_nb_face = buffer[2];
    m_nb_cell = buffer[3];
    m_edge_allocated = buffer[4];
    m_face_allocated = buffer[5];
    m_cell_allocated = buffer[6];
    m_index = buffer[7];
    m_nb_reference = buffer[8];
    //! AMR
    if (buf_size>=13){
      m_nb_hParent = buffer[9];
      m_nb_hChildren = buffer[10];
      m_hParent_allocated = buffer[11];
      m_hChild_allocated = buffer[12];
    }
  }
  else if (buf_size>=4){
    m_index = buffer[2];
    m_nb_reference = buffer[3];
  }

  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Nombre d'informations à écrire pour les données.
 *
 * A partir de version 3.6, on ne conserve plus d'informations sur le
 * nombre d'entités connectées.
 */
Integer ItemSharedInfo::
serializeWriteSize()
{
  return 6;
}

Integer ItemSharedInfo::
serializeSize()
{
	return serializeWriteSize();
}

Integer ItemSharedInfo::
serializeAMRSize()
{
  return serializeWriteSize();
}

Integer ItemSharedInfo::
serializeNoAMRSize()
{
  return serializeWriteSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
serializeWrite(Int32ArrayView buffer)
{
  buffer[0] = m_type_id; // Doit toujours être le premier

  buffer[1] = 0x0307; // Numéro de version (3.7).

  buffer[2] = m_index;
  buffer[3] = m_nb_reference;

  buffer[4] = 0; // Réservé
  buffer[5] = 0; // Réservé
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
  ARCANE_ASSERT(m_hParent_allocated==0,("m_hParent_allocated should be zero"));
  ARCANE_ASSERT(m_hChild_allocated==0,("m_hChild_allocated should be zero"));
  ARCANE_ASSERT(m_nb_edge==0,("m_edge should be zero"));
  ARCANE_ASSERT(m_nb_face==0,("m_face should be zero"));
  ARCANE_ASSERT(m_nb_cell==0,("m_cell should be zero"));
  ARCANE_ASSERT(m_edge_allocated==0,("m_edge_allocated should be zero"));
  ARCANE_ASSERT(m_face_allocated==0,("m_face_allocated should be zero"));
  ARCANE_ASSERT(m_cell_allocated==0,("m_cell_allocated should be zero"));

  bool is_amr_activated = false;

  if (ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell){
    IItemFamily* base_family = m_item_family;
    m_nb_parent = 0;
    if (base_family)
      m_nb_parent = base_family->parentFamilyDepth();
    ARCANE_ASSERT((m_nb_parent<=1),("More than one parent level: not implemented"));
    is_amr_activated = m_items->mesh->isAmrActivated();
  }

  m_first_node = FIRST_NODE_INDEX;
  m_first_edge = m_first_node + m_nb_node;
  m_first_face = m_first_edge;
  m_first_cell = m_first_edge;
  m_first_parent = m_first_edge;

  //! AMR
  if (is_amr_activated){
    m_first_hParent = m_first_edge + m_nb_parent;
    m_first_hChild = m_first_edge + m_nb_parent;
  }

  m_needed_memory = m_first_edge + m_nb_parent;
  m_minimum_needed_memory = COMMON_BASE_MEMORY + m_nb_node + m_nb_parent;
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

