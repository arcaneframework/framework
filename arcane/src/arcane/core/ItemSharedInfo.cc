﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfo.cc                                           (C) 2000-2023 */
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
#include "arcane/ItemInfoListView.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo ItemSharedInfo::nullItemSharedInfo;

// TODO: A terme il faudra pouvoir changer cela pour utiliser une valeur
// allouée dynamiquement ce qui permettra à cette instance d'être utilisée
// sur GPU.
ItemSharedInfo* ItemSharedInfo::nullItemSharedInfoPointer = &ItemSharedInfo::nullItemSharedInfo;

namespace
{
// Suppose NULL_ITEM_UNIQUE_ID == (-1) et NULL_ITEM_LOCAL_ID == (-1)
// Cree un pseudo-tableau qui pourra etre indexé avec NULL_ITEM_LOCAL_ID
// pour la maille nulle.
Int64 null_int64_buf[2] = { NULL_ITEM_UNIQUE_ID, NULL_ITEM_UNIQUE_ID };
Int64ArrayView null_unique_ids(1,null_int64_buf + 1);

Int32 null_parent_items_buf[2] = { NULL_ITEM_ID, NULL_ITEM_ID };
Int32ArrayView null_parent_item_ids(1,null_parent_items_buf+1);

Int32 null_owners_buf[2] = { A_NULL_RANK, A_NULL_RANK };
Int32ArrayView null_owners(1,null_owners_buf);

Int32 null_flags_buf[2] = { 0, 0 };
Int32ArrayView null_flags(1,null_flags_buf+1);

Int16 null_type_ids_buf[2] = { IT_NullType, IT_NullType };
Int16ArrayView null_type_ids(1,null_type_ids_buf + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo()
: m_connectivity(&ItemInternalConnectivityList::nullInstance)
, m_unique_ids(null_unique_ids)
, m_parent_item_ids(null_parent_item_ids)
, m_owners(null_owners)
, m_flags(null_flags)
, m_type_ids(null_type_ids)
{
  _init(IK_Unknown);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo::
ItemSharedInfo(IItemFamily* family,MeshItemInternalList* items,ItemInternalConnectivityList* connectivity)
: m_items(items)
, m_connectivity(connectivity)
, m_item_family(family)
, m_item_type_mng(family->mesh()->itemTypeMng())
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

ItemInternal* ItemSharedInfo::
_parent(Int32 id) const
{
  // En pointant vers le bon champ du MeshItemInternalList dans le maillage parent
  // TODO GG: on pourrait conserver une fois pour toute l'instance de 'ItemInfoListView'
  return ItemCompatibility::_itemInternal(m_items->mesh->itemFamily(m_item_kind)->parentFamily()->itemInfoListView()[id]);
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
_setParentV2(Int32 local_id,[[maybe_unused]] Integer aindex,Int32 parent_local_id)
{
  ARCANE_ASSERT((aindex==0),("Only one parent access implemented"));
  m_parent_item_ids[local_id] = parent_local_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32* ItemSharedInfo::
_parentPtr(Int32 local_id)
{
  // GG: ATTENTION: Cela ne fonctionne que si on a au plus un parent.
  return m_parent_item_ids.ptrAt(local_id);
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

ItemInternalVectorView ItemSharedInfo::
nodes(Int32) const
{
  return ItemInternalVectorView();
}

ItemInternalVectorView ItemSharedInfo::
edges(Int32) const
{
  return ItemInternalVectorView();
}

ItemInternalVectorView ItemSharedInfo::
faces(Int32) const
{
  return ItemInternalVectorView();
}

ItemInternalVectorView ItemSharedInfo::
cells(Int32) const
{
  return ItemInternalVectorView();
}

ItemInternalVectorView ItemSharedInfo::
hChildren(Int32) const
{
  return ItemInternalVectorView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfo::
updateMeshItemInternalList()
{
  m_connectivity->updateMeshItemInternalList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

