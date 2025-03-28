// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivitySynchronizer.cc                             (C) 2000-2024 */
/*                                                                           */
/* Synchronization des connectivités.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemConnectivitySynchronizer.h"
#include "arcane/mesh/IItemConnectivityGhostPolicy.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemConnectivitySynchronizer::
ItemConnectivitySynchronizer(IItemConnectivity* connectivity,
                             IItemConnectivityGhostPolicy* ghost_policy)
: m_connectivity(connectivity)
, m_ghost_policy(ghost_policy)
, m_parallel_mng(m_connectivity->targetFamily()->mesh()->parallelMng())
, m_added_ghost(m_parallel_mng->commSize())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySynchronizer::
synchronize()
{
  // Compute ghost element for ToFamily
  mesh::ExtraGhostItemsManager extra_ghost_mng(this);
  extra_ghost_mng.addExtraGhostItemsBuilder(this);
  extra_ghost_mng.computeExtraGhostItems();
}

void ItemConnectivitySynchronizer::
computeExtraItemsToSend()
{
  // Get, for each rank k, ToFamily item that have to be shared with rank k, the
  m_data_to_send.clear();
  Int32ConstArrayView ranks = m_ghost_policy->communicatingRanks();
  m_data_to_send.resize(m_parallel_mng->commSize());
  const Int32 my_rank = m_parallel_mng->commRank();
  for (Integer i = 0; i < ranks.size(); ++i){
    Integer rank = ranks[i];
    Int32Array& data_to_send = m_data_to_send[rank];
    Int32SharedArray shared_items = m_ghost_policy->sharedItems(rank,m_connectivity->targetFamily()->name());
    Int32SharedArray shared_items_connected_items = m_ghost_policy->sharedItemsConnectedItems(rank,m_connectivity->sourceFamily()->name());
    _getItemToSend(shared_items,shared_items_connected_items, rank);
    data_to_send.add(shared_items.size());
    data_to_send.addRange(shared_items);
    data_to_send.addRange(shared_items_connected_items);
    data_to_send.addRange(my_rank,shared_items.size()); // shared item owner
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySynchronizer::
serializeGhostItems(ISerializer* buffer,Int32ConstArrayView ghost_item_info)
{
  // Treat ghost item info : split into item_uids, nb_dof_per_item, dof_uids
  Integer nb_shared_item = ghost_item_info[0];
  Int32ConstArrayView shared_item_lids = ghost_item_info.subConstView(1,nb_shared_item);
  Int32ConstArrayView shared_item_connected_item_lids = ghost_item_info.subConstView(nb_shared_item+1,nb_shared_item);
  Int32ConstArrayView shared_item_owners = ghost_item_info.subConstView(2*nb_shared_item+1,nb_shared_item);
  // Convert in uids (items must belong to current process)
  Int64SharedArray shared_item_uids(nb_shared_item);
  Int64SharedArray shared_item_connected_item_uids(nb_shared_item);
  Integer i = 0, j= 0;
  ENUMERATE_ITEM(item,m_connectivity->targetFamily()->view(shared_item_lids)){
    shared_item_uids[i++] = item->uniqueId().asInt64();
  }
  ENUMERATE_ITEM(item,m_connectivity->sourceFamily()->view(shared_item_connected_item_lids)){
    shared_item_connected_item_uids[j++] = item->uniqueId().asInt64();
  }

  // Serialize item_uid, nb_dof, shared_item_connected_item_uids
  buffer->setMode(ISerializer::ModeReserve);
  buffer->reserveInt64(1);
  buffer->reserveSpan(shared_item_uids);
  buffer->reserveSpan(shared_item_connected_item_uids);
  buffer->reserveSpan(shared_item_owners);

  buffer->allocateBuffer();
  buffer->setMode(ISerializer::ModePut);

  buffer->putInt64(nb_shared_item);
  buffer->putSpan(shared_item_uids);
  buffer->putSpan(shared_item_connected_item_uids);
  buffer->putSpan(shared_item_owners);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySynchronizer::
addExtraGhostItems (ISerializer* buffer)
{
  // Deserialiser le buffer et faire un addGhost classique, puis faire la connexion grace a l'item property
  buffer->setMode(ISerializer::ModeGet);

  Int64 nb_shared_items = buffer->getInt64();

  Int64SharedArray shared_item_uids(nb_shared_items);
  Int64SharedArray shared_item_connected_items_uids(nb_shared_items);
  Int32SharedArray shared_items_owners(nb_shared_items);

  buffer->getSpan(shared_item_uids);
  buffer->getSpan(shared_item_connected_items_uids);
  buffer->getSpan(shared_items_owners);

  // Remove possible repetition in shared_items and impact owners. Necessary to add these items in the family
  Int64SharedArray   shared_item_uids_to_add(shared_item_uids);
  IntegerSharedArray shared_item_owners_to_add(shared_items_owners);
  _removeDuplicatedValues(shared_item_uids_to_add,shared_item_owners_to_add);

  // Add shared items
  Int32SharedArray shared_item_lids_added(shared_item_uids_to_add.size());
  m_connectivity->targetFamily()->addGhostItems(shared_item_uids_to_add,shared_item_lids_added,shared_item_owners_to_add);
  m_connectivity->targetFamily()->endUpdate();

  // update connectivity
  Int32SharedArray shared_item_lids(nb_shared_items);
  bool do_fatal = true; // shared_items have been adde to the family
  m_connectivity->targetFamily()  ->itemsUniqueIdToLocalId(shared_item_lids,shared_item_uids,do_fatal);
  // shared_item_connected_item are not compulsory present on the new subdomain where shared_items have been send.
  // Send shared_item_connected_items_uids to GhostPolicy where they will be handled (possibly converted to lid if they are present).
  m_ghost_policy->updateConnectivity(shared_item_lids,shared_item_connected_items_uids);

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySynchronizer::
_getItemToSend(Int32SharedArray& shared_items, Int32SharedArray& shared_items_connected_items, const Integer rank)
{
  // remove null items.
  Int32UniqueArray non_null_shared_items, non_null_shared_items_connected_items;
  non_null_shared_items.reserve(shared_items.size());
  non_null_shared_items_connected_items.reserve(shared_items_connected_items.size());
  auto i = 0;
  for (auto item : shared_items) {
    if (item != NULL_ITEM_LOCAL_ID) {
      non_null_shared_items.add(shared_items[i]);
      non_null_shared_items_connected_items.add(shared_items_connected_items[i]);
    }
    ++i;
  }
  shared_items.copy(non_null_shared_items);
  shared_items_connected_items.copy(non_null_shared_items_connected_items);
  // Filter: don't add ghost twice. The synchronizer remember the ghost added and don't add them twice.
  Int64SharedArray shared_items_uids(shared_items.size());
  ItemVectorView shared_items_view = m_connectivity->targetFamily()->view(shared_items);
  for (Integer i = 0; i < shared_items_view.size(); ++i)
    shared_items_uids[i] = shared_items_view[i].uniqueId().asInt64();
  std::set<Int64>& added_ghost = m_added_ghost[rank];


 // reverse order to handle index changes due to remove
  for (Integer i = shared_items_uids.size()-1; i >=0  ; --i ){
    if (!added_ghost.insert(shared_items_uids[i]).second){
      // check if already added ghost
      shared_items.remove(i);
      shared_items_connected_items.remove(i);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivitySynchronizer::
_removeDuplicatedValues(Int64SharedArray& shared_item_uids, IntegerSharedArray& shared_item_owners)
{
  // Reverse order for the loop : otherwise wrong !! (since elements are removed and thus index are modified)
  std::set<Int64> shared_uids_set;
  for (Integer i = shared_item_uids.size()-1; i >= 0 ; --i){
    if (! shared_uids_set.insert(shared_item_uids[i]).second){
      // element already in the set, remove it from both arrays.
      shared_item_uids.remove(i);
      shared_item_owners.remove(i);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* ItemConnectivitySynchronizer::
subDomain()
{
  return m_connectivity->targetFamily()->mesh()->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
