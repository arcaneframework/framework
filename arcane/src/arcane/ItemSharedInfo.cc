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
#include "arcane/utils/FatalErrorException.h"

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
ItemSharedInfo(IItemFamily* family,MeshItemInternalList* items,
               ItemInternalConnectivityList* connectivity,ItemVariableViews* variable_views)
: m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_item_type_mng(family->mesh()->itemTypeMng())
, m_unique_ids(&(variable_views->m_unique_ids_view))
, m_parent_item_ids(&(variable_views->m_parent_ids_view))
, m_owners(&(variable_views->m_owners_view))
, m_flags(&(variable_views->m_flags_view))
, m_type_ids(&(variable_views->m_type_ids_view))
, m_item_kind(family->itemKind())
{
  _init(m_item_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
print(std::ostream& o) const
{
  o << " This: " << this
    << " NbParent; " << m_nb_parent
    << " Items: " << m_items
    << " Connectivity: " << m_connectivity
    << " Family: " << m_item_family->fullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
_init(eItemKind ik)
{
  if (ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell){
    IItemFamily* base_family = m_item_family;
    m_nb_parent = 0;
    if (base_family)
      m_nb_parent = base_family->parentFamilyDepth();
    ARCANE_ASSERT((m_nb_parent<=1),("More than one parent level: not implemented"));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalArrayView ItemSharedInfo::
_parents() const
{
  // En pointant vers le bon champ du MeshItemInternalList dans le maillage parent
  return m_items->mesh->itemFamily(m_item_kind)->parentFamily()->itemsInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
setNode(Int32,Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setEdge(Int32,Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setFace(Int32,Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setCell(Int32,Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setHParent(Int32,Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setHChild(Int32,Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

ItemInternal* ItemSharedInfo::
parent(Integer,Integer) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setParent(Integer,Integer,Integer) const
{
  ARCANE_FATAL("This method is no longer valid");
}

Int32 ItemSharedInfo::
owner(Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setOwner(Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

Int32 ItemSharedInfo::
flags(Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
setFlags(Int32,Int32) const
{
  ARCANE_FATAL("This method is no longer valid");
}

void ItemSharedInfo::
_setInfos(Int32*)
{
  ARCANE_FATAL("This method is no longer valid");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
_setParentV2(Int32 local_id,[[maybe_unused]] Integer aindex,Int32 parent_local_id) const
{
  ARCANE_ASSERT((aindex==0),("Only one parent access implemented"));
  (*m_parent_item_ids)[local_id] = parent_local_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32* ItemSharedInfo::
_parentPtr(Int32 local_id) const
{
  // GG: ATTENTION: Cela ne fonctionne que si on a au plus un parent.
  return m_parent_item_ids->ptrAt(local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeInfo* ItemSharedInfo::
typeInfoFromId(Int32 type_id) const
{
  return m_item_type_mng->typeFromId(type_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemSharedInfo::
typeId() const
{
  ARCANE_FATAL("This method is no longer valid");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

