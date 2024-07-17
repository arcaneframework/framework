// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFFamily.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Famille de degre de liberte                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/mesh/DoFFamily.h"

#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/ItemTypeMng.h"
#include "arcane/ItemTypeInfo.h"
#include "arcane/IExtraGhostItemsBuilder.h"

#include "arcane/mesh/ExtraGhostItemsManager.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DoFFamily::
DoFFamily(IMesh* mesh, const String& name)
  : ItemFamily(mesh,IK_DoF,name)
  , m_shared_info(0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFFamily::
build()
{
  ItemFamily::build();
  m_sub_domain_id = subDomain()->subDomainId();
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  ItemTypeInfo* dof_type_info = itm->typeFromId(IT_NullType);
  m_shared_info = _findSharedInfo(dof_type_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DoFVectorView
DoFFamily::
addDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids)
{
  ARCANE_ASSERT((dof_uids.size() == dof_lids.size()),("in addDofs(uids,lids) given uids and lids array must have same size"))
  _addItems(dof_uids,dof_lids);
  return view(dof_lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DoFVectorView
DoFFamily::
addGhostDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids, Int32ConstArrayView owners)
{
  ARCANE_ASSERT((dof_uids.size() == dof_lids.size() && (dof_uids.size() == owners.size())),("in addGhostDofs given uids, lids and owners array must have same size"))
  addGhostItems(dof_uids,dof_lids,owners);
  return view(dof_lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFFamily::
_addItems(Int64ConstArrayView unique_ids, Int32ArrayView items)
{
  Integer nb_item = unique_ids.size();
   if (nb_item==0)
     return;
   preAllocate(nb_item);
   for( Integer i=0; i<nb_item; ++i ){
     Int64 uid = unique_ids[i];
     ItemInternal* ii = _allocDoF(uid);
     items[i] = ii->localId();
   }

   m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFFamily::
addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items, Int32ConstArrayView owners)
{
  Integer nb_item = unique_ids.size();
  if (nb_item==0)
    return;
  preAllocate(nb_item);
  for( Integer i=0; i<nb_item; ++i ){
    Int64 uid   = unique_ids[i];
    Int32 owner = owners[i];
    ItemInternal* ii = _allocDoFGhost(uid,owner);
    items[i] = ii->localId();
  }

  m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFFamily::
internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost)
{
  ARCANE_UNUSED(keep_ghost);
  _removeMany(local_ids);

  m_need_prepare_dump = true;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFFamily::
removeDoFs(Int32ConstArrayView items_local_id)
{
  internalRemoveItems(items_local_id,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFFamily::
computeSynchronizeInfos()
{
  debug() << "Creating the list of ghosts dofs";
  ItemFamily::computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFFamily::
_printInfos(Integer nb_added)
{
  Integer nb_in_map = 0;
  for( auto i : itemsMap().buckets() ){
    for( ItemInternalMap::Data* nbid = i; nbid; nbid = nbid->next() ){
      ++nb_in_map;
    }
  }

  info() << "DoFFamily: added=" << nb_added
         << " nb_internal=" << infos().m_internals.size()
         << " nb_free=" << infos().m_free_internals.size()
         << " map_nb_bucket=" << itemsMap().buckets().size()
         << " map_size=" << nb_in_map;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFFamily::
preAllocate(Integer nb_item)
{
  // Copy paste de particle, pas utilise pour l'instant
  Integer nb_hash = itemsMap().buckets().size();
  Integer wanted_size = 2*(nb_item+infos().nbItem());
  if (nb_hash<wanted_size)
    itemsMap().resize(wanted_size,true);

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* DoFFamily::
_allocDoF(const Int64 uid)
{
  bool need_alloc; // given by alloc
  //ItemInternal* item_internal = ItemFamily::_allocOne(uid,need_alloc);
  ItemInternal* item_internal = ItemFamily::_findOrAllocOne(uid,need_alloc);
  if (!need_alloc)
    item_internal->setUniqueId(uid);
  else{
    _allocateInfos(item_internal,uid,m_shared_info);
  }
  // Un dof appartient de base au sous-domaine qui l'a créé (sauf ghost)
  item_internal->setOwner(m_sub_domain_id,m_sub_domain_id);
  return item_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* DoFFamily::
_allocDoFGhost(const Int64 uid, const Int32 owner)
{
  bool need_alloc; // given by alloc
  //ItemInternal* item_internal = ItemFamily::_allocOne(uid,need_alloc);
  ItemInternal* item_internal = ItemFamily::_findOrAllocOne(uid,need_alloc);
  //ItemInternal* item_internal = m_infos.findOrAllocOne(uid,need_alloc);
  if (!need_alloc)
    item_internal->setUniqueId(uid);
  else{
    _allocateInfos(item_internal,uid,m_shared_info);
  }
  // Une particule appartient toujours au sous-domaine qui l'a créée
  item_internal->setOwner(owner,m_sub_domain_id);
  return item_internal;
}

ItemInternal* DoFFamily::
_findOrAllocDoF(const Int64 uid,bool& is_alloc)
{
  ItemInternal* item_internal = ItemFamily::_findOrAllocOne(uid,is_alloc);
  if (!is_alloc) {
    item_internal->setUniqueId(uid);
  }
  else {
    _allocateInfos(item_internal,uid,m_shared_info);
    // Un dof appartient de base au sous-domaine qui l'a créé (sauf ghost)
    item_internal->setOwner(m_sub_domain_id,m_sub_domain_id);
  }
  return item_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
