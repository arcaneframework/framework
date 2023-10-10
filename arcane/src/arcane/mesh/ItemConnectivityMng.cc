// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityMng.cc                                      (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des connectivités des entités.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <algorithm>

#include "arcane/mesh/ItemConnectivityMng.h"
#include "arcane/mesh/ItemConnectivitySynchronizer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemConnectivitySynchronizer* ItemConnectivityMng::
createSynchronizer(IItemConnectivity* connectivity, IItemConnectivityGhostPolicy* connectivity_ghost_policy)
{
  std::map<IItemConnectivity*, IItemConnectivitySynchronizer*>::iterator ite = m_synchronizers.find(connectivity);
  IItemConnectivitySynchronizer* synchronizer = NULL;
  if (ite == m_synchronizers.end()){
    synchronizer = new ItemConnectivitySynchronizer(connectivity,connectivity_ghost_policy);
    m_synchronizers[connectivity] = synchronizer;
  }
  else{
    // Create a new synchronizer for this connectivity
    delete ite->second;
    synchronizer = new ItemConnectivitySynchronizer(connectivity,connectivity_ghost_policy);
    m_synchronizers.erase(ite);
    m_synchronizers[connectivity] = synchronizer;
  }
  return synchronizer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
_register(const String& connectivity_name, const String& source_family_name, const String& target_family_name)
{
  std::pair<FamilyStateMap::iterator,bool> source_family_state_insertor = m_family_states.insert(std::make_pair(source_family_name,FamilyState()));
  std::pair<FamilyStateMap::iterator,bool> target_family_state_insertor = m_family_states.insert(std::make_pair(target_family_name,FamilyState()));
  std::pair<ConnectivityStateMap::iterator,bool> connectivity_state_insertor = m_connectivity_states.insert(std::make_pair(connectivity_name,ConnectivityState()));
  FamilyState& source_family_state = source_family_state_insertor.first->second;
  FamilyState& target_family_state = target_family_state_insertor.first->second;
  ConnectivityState& connectivity_state = connectivity_state_insertor.first->second;
  // If insertion occurs for connectivity, set its state to the family current state
  if (connectivity_state_insertor.second){
    connectivity_state.m_state_with_source_family.m_last_family_state = source_family_state.m_state;
    connectivity_state.m_state_with_source_family.m_last_added_item_index = source_family_state.m_added_items.size()-1;
    connectivity_state.m_state_with_source_family.m_last_removed_item_index = source_family_state.m_removed_items.size()-1;
    connectivity_state.m_state_with_target_family.m_last_family_state = target_family_state.m_state;
    connectivity_state.m_state_with_target_family.m_last_added_item_index = target_family_state.m_added_items.size()-1;
    connectivity_state.m_state_with_target_family.m_last_removed_item_index = target_family_state.m_removed_items.size()-1;
  }
  else
    m_trace_mng->warning() << "Connectivity " << connectivity_name << " already registered.";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
setModifiedItems(IItemFamily* family, Int32ConstArrayView added_items,Int32ConstArrayView removed_items)
{
  FamilyState& family_state = _findFamily(family->fullName());

  // Add added items
  family_state.m_added_items.addRange(added_items);
  // Add removed items
  family_state.m_removed_items.addRange(removed_items);
  // Increment family state
  ++family_state.m_state;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
getSourceFamilyModifiedItems(IItemConnectivity* connectivity, Int32ArrayView& added_items, Int32ArrayView& removed_items)
{
  ConnectivityState& connectivity_state = _findConnectivity(connectivity->name());
  FamilyState& family_state = _findFamily(connectivity->sourceFamily()->fullName());
  _getModifiedItems(connectivity_state.m_state_with_source_family,family_state,added_items,removed_items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
getTargetFamilyModifiedItems(IItemConnectivity* connectivity, Int32ArrayView& added_items, Int32ArrayView& removed_items)
{
  ConnectivityState& connectivity_state = _findConnectivity(connectivity->name());
  FamilyState& family_state = _findFamily(connectivity->targetFamily()->fullName());
  _getModifiedItems(connectivity_state.m_state_with_target_family,family_state,added_items,removed_items);
}

void ItemConnectivityMng::
getSourceFamilyModifiedItems(IIncrementalItemConnectivity* connectivity, Int32ArrayView& added_items, Int32ArrayView& removed_items)
{
  ConnectivityState& connectivity_state = _findConnectivity(connectivity->name());
  FamilyState& family_state = _findFamily(connectivity->sourceFamily()->fullName());
  _getModifiedItems(connectivity_state.m_state_with_source_family,family_state,added_items,removed_items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
getTargetFamilyModifiedItems(IIncrementalItemConnectivity* connectivity, Int32ArrayView& added_items, Int32ArrayView& removed_items)
{
  ConnectivityState& connectivity_state = _findConnectivity(connectivity->name());
  FamilyState& family_state = _findFamily(connectivity->targetFamily()->fullName());
  _getModifiedItems(connectivity_state.m_state_with_target_family,family_state,added_items,removed_items);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
_getModifiedItems(ConnectivityStateData& connectivity_state, FamilyState& family_state,
                  Int32ArrayView& added_items, Int32ArrayView& removed_items)
{
  if (connectivity_state.m_last_family_state == family_state.m_state)
    return; // Connectivity is up to date

  Integer first_added_item_index = connectivity_state.m_last_added_item_index +1;
  Integer nb_added_items = family_state.m_added_items.size()-first_added_item_index;
  auto unfiltered_added_items = family_state.m_added_items.subView(first_added_item_index,nb_added_items);
  // Filter null lid (may occur if both add and remove actions occur before calling this method)
  family_state.m_current_added_items.clear();
  family_state.m_current_added_items.resize(nb_added_items);
  auto true_size = 0;
  std::copy_if(unfiltered_added_items.begin(),unfiltered_added_items.end(),
               family_state.m_current_added_items.begin(),
               [&true_size](Int32 const& item_lid){
                   auto do_copy = (item_lid != NULL_ITEM_LOCAL_ID);
                   if (do_copy) ++true_size;
                   return do_copy;
               });
  family_state.m_current_added_items.resize(true_size);
  added_items = family_state.m_current_added_items.view();

  Integer first_removed_item_index = connectivity_state.m_last_removed_item_index +1;
  Integer nb_removed_items = family_state.m_removed_items.size()-first_removed_item_index;
  removed_items = family_state.m_removed_items.subView(first_removed_item_index,nb_removed_items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
setUpToDate(IItemConnectivity* connectivity)
{
  ConnectivityState& connectivity_state = _findConnectivity(connectivity->name());
  FamilyState& source_family_state = _findFamily(connectivity->sourceFamily()->fullName());
  FamilyState& target_family_state = _findFamily(connectivity->targetFamily()->fullName());
  _setUpToDate(connectivity_state.m_state_with_source_family,source_family_state);
  _setUpToDate(connectivity_state.m_state_with_target_family,target_family_state);
}

void ItemConnectivityMng::
setUpToDate(IIncrementalItemConnectivity* connectivity)
{
  ConnectivityState& connectivity_state = _findConnectivity(connectivity->name());
  FamilyState& source_family_state = _findFamily(connectivity->sourceFamily()->fullName());
  FamilyState& target_family_state = _findFamily(connectivity->targetFamily()->fullName());
  _setUpToDate(connectivity_state.m_state_with_source_family,source_family_state);
  _setUpToDate(connectivity_state.m_state_with_target_family,target_family_state);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
_setUpToDate(ConnectivityStateData& connectivity_state, FamilyState& family_state)
{
  connectivity_state.m_last_family_state = family_state.m_state;
  connectivity_state.m_last_added_item_index   = family_state.m_added_items.size()-1;
  connectivity_state.m_last_removed_item_index = family_state.m_removed_items.size()-1;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemConnectivityMng::
_lastUpdateSourceFamilyState(const String& connectivity_name)
{
  return _findConnectivity(connectivity_name).m_state_with_source_family.m_last_family_state;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemConnectivityMng::
_lastUpdateTargetFamilyState(const String& connectivity_name)
{
  return _findConnectivity(connectivity_name).m_state_with_target_family.m_last_family_state;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemConnectivityMng::
_familyState(const String& family_full_name)
{
  return _findFamily(family_full_name).m_state;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConnectivityState& ItemConnectivityMng::
_findConnectivity(const String& connectivity_name)
{
  ConnectivityStateMap::iterator connectivity_state_iterator =  m_connectivity_states.find(connectivity_name);
  if (connectivity_state_iterator == m_connectivity_states.end())
    throw FatalErrorException(String::format(
      "Cannot find connectivity {0}. Use registerConnectivity(connectivity) first",connectivity_name));
  return connectivity_state_iterator->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FamilyState& ItemConnectivityMng::
_findFamily(const String& family_full_name)
{
  FamilyStateMap::iterator family_state_iterator =  m_family_states.find(family_full_name);
  if (family_state_iterator == m_family_states.end())
    throw FatalErrorException(String::format(
        "Cannot find family {0}. Use registerConnectivity(connectivity) first",family_full_name));
  return family_state_iterator->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityMng::
notifyLocalIdChanged(IItemFamily* family, Int32ConstArrayView old_to_new_ids, Integer nb_item)
{
  // Rk this is a temporary patch, connectivity update has to be refactored !
  FamilyState& family_state = _findFamily(family->fullName());
  // Treat added items
  Integer new_id;
  for (Arcane::Integer i = 0; i < family_state.m_added_items.size(); ++i)
    {
      if (family_state.m_added_items[i] != NULL_ITEM_LOCAL_ID)
        {
          new_id = old_to_new_ids[family_state.m_added_items[i]];
          if (new_id < nb_item) family_state.m_added_items[i] = new_id;// OK valid item
          else family_state.m_added_items[i] = NULL_ITEM_LOCAL_ID;// KO invalid item. old_to_new has no null_item_lid. it has lid > nb_item. These lids will be removed in finish compact items...
        }
    }
  // Clear removed items, connectivity update is automatically launched by compaction
  family_state.m_removed_items.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
