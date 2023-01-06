// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "arcane/mesh/ItemGroupDynamicMeshObserver.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/ItemFamily.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/ItemPrinter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupDynamicMeshObserver::
executeExtend(const Int32ConstArrayView * new_items_info)
{
  IItemFamily * parent_family = m_mesh->parentGroup().itemFamily();

  // Les familles parents ont déjà été branchées.
  ARCANE_ASSERT((m_mesh->cellFamily()->parentFamily()->itemKind() == parent_family->itemKind()),
                ("Bad Face parent family %s",m_mesh->cellFamily()->parentFamily()->name().localstr()));

  ItemVectorView all_group(parent_family->view(*new_items_info));
  
  m_mesh->incrementalBuilder()->addParentItems(all_group,IK_Cell);
}

/*---------------------------------------------------------------------------*/

void ItemGroupDynamicMeshObserver::
executeReduce(const Int32ConstArrayView * info)
{
  IItemFamily * family = m_mesh->cellFamily();
  IItemFamily * parent_family = m_mesh->parentGroup().itemFamily();
  ItemInfoListView parent_internals(parent_family);

  // info contient les localIds supprimés de l'ancien groupe,
  const Int32ConstArrayView & parent_deleted_lids = *info;
    
  // Les uniqueIds des Cell est identique au parent donc on cherche dans la famille locale
  Int64UniqueArray parent_uids; 
  parent_uids.reserve(parent_deleted_lids.size());
  for(Integer i=0;i<parent_deleted_lids.size();++i)
  {
    Int64 parent_uid = parent_internals[parent_deleted_lids[i]]->uniqueId();
    if (parent_uid != NULL_ITEM_UNIQUE_ID) // prevent from completly detached items
    {
      parent_uids.add(parent_uid);
    }
  }  


  Int32UniqueArray to_delete_lids(parent_uids.size(),NULL_ITEM_LOCAL_ID);
  family->itemsUniqueIdToLocalId(to_delete_lids,
                                 parent_uids,
                                 true); // Fatal si non trouvé

#ifdef ARCANE_DEBUG
  ITraceMng * traceMng = m_mesh->traceMng();
  ItemInfoListView internals(family);
  for(Integer i=0;i<parent_uids.size();++i) {
    traceMng->debug(Trace::High) << "Reduce with item " 
                                 << ItemPrinter(internals[to_delete_lids[i]])
                                 << " parent item " << parent_uids[i]; 
  }
#endif /* ARCANE_DEBUG */

  m_mesh->removeCells(to_delete_lids);
}

/*---------------------------------------------------------------------------*/

void ItemGroupDynamicMeshObserver::
executeCompact(const Int32ConstArrayView * pinfo)
{
  if (!pinfo)
    throw ArgumentException(A_FUNCINFO,"Compact info required");

  // Le compactage est traité dans ItemFamily::_compactFromParentFamily
  // afin de procéder de manière consistante famille par famille
  if (arcaneIsDebug()){
    ITraceMng* trace_mng = m_mesh->traceMng();
    const Int32ConstArrayView& old_to_new_ids = *pinfo;
    IItemFamily* family = m_mesh->cellFamily();
    ItemFamily* parent_family = ARCANE_CHECK_POINTER(dynamic_cast<ItemFamily*>(family->parentFamily()));
    for(Integer i=0; i<old_to_new_ids.size() ; ++i ) {
      trace_mng->debug(Trace::Highest) << "OldToNew " << parent_family->name() << " " << i << " " << old_to_new_ids[i];
    }
  }
}

/*---------------------------------------------------------------------------*/

void ItemGroupDynamicMeshObserver::
executeInvalidate()
{
  ARCANE_FATAL("Not implemented");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
