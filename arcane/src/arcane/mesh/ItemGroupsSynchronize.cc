// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSynchronize.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Group synchronizations.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemGroupsSynchronize.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Math.h"

#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/mesh/CommonItemGroupFilterer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupsSynchronize::
ItemGroupsSynchronize(IItemFamily* item_family)
: TraceAccessor(item_family->traceMng())
, m_item_family(item_family)
, m_var(VariableBuildInfo(item_family,"MeshItemGroupSynchronize",
                          IVariable::PNoDump|IVariable::PNoRestore),
        item_family->itemKind())
{
  _setGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupsSynchronize::
ItemGroupsSynchronize(IItemFamily* item_family,ItemGroupCollection groups)
: TraceAccessor(item_family->traceMng())
, m_item_family(item_family)
, m_var(VariableBuildInfo(item_family,"MeshItemGroupSynchronize",
                          IVariable::PNoDump|IVariable::PNoRestore),
        item_family->itemKind())
, m_groups(groups)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupsSynchronize::
~ItemGroupsSynchronize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupsSynchronize::
_setGroups()
{
  // Constructing the list of groups
  m_groups.clear();
  for( const ItemGroup& group : m_item_family->groups() ){
    if (!group.internal()->needSynchronization())
      continue;
    m_groups.add(group);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupsSynchronize::
synchronize()
{
  ItemGroup all_items = m_item_family->allItems();

  // Constructing the list of groups
  UniqueArray<ItemGroup> groups;
  CommonItemGroupFilterer group_filterer(m_item_family);
  for( ItemGroup group : m_groups ){
    ARCANE_ASSERT((!group.null()),("Null group in ItemGroupsSynchronize"));
    groups.add(group);
    group_filterer.addGroupToFilter(group);
  }
  // TODO: check if it shouldn't be done every time
  // This way we are sure that the groups are properly sorted
  if (arcaneIsCheck())
    group_filterer.applyFiltering();

  Int32UniqueArray group_items; // Array storing the list of evolving items
  group_items.reserve(m_item_family->maxLocalId());

  const Integer max_aggregate_size = 
    sizeof(IntAggregator)*8-1; // To stay within the positive integer space
  const Integer aggregate_count = 
    (groups.size() / max_aggregate_size) +
    ((groups.size() % max_aggregate_size)?1:0);

  for(Integer i_aggregate=0; i_aggregate < aggregate_count; ++i_aggregate) {
    const Integer first_group = i_aggregate * max_aggregate_size;
    const Integer current_aggregate_size = math::min(max_aggregate_size, 
                                                     groups.size()-first_group);
    m_var.fill(0); // Initialization of the parallel array
    for(Integer i_group=0;i_group<current_aggregate_size;++i_group) {
      const IntAggregator current_mask = static_cast<IntAggregator>(1)<<i_group;
      ItemGroup group = groups[first_group+i_group];
      ENUMERATE_ITEM(iitem,group){
        m_var[iitem] |= current_mask;
      }
    }

    // Sharing group content info on the aggregate
    m_var.synchronize();

    for(Integer i_group=0;i_group<current_aggregate_size;++i_group) {
      const IntAggregator current_mask = static_cast<IntAggregator>(1)<<i_group;
      ItemGroup group = groups[first_group+i_group];
      const Integer current_size = group.size();
      if (group.internal()->hasInfoObserver()) {
        // Switching to incremental mode
        // Searching for missing items
        group_items.clear();
        ENUMERATE_ITEM(iitem, group) {
          if ((m_var[iitem] & current_mask) == 0) {
            group_items.add(iitem.itemLocalId());
          } else {
            m_var[iitem] &= ~current_mask; // marks as already existing in the group
          }
        }
        Integer removed_size = group_items.size();
        group.removeItems(group_items);
        // Searching for new items
        group_items.clear();
        ENUMERATE_ITEM(iitem, all_items) {
          if ((m_var[iitem] & current_mask) != 0) {
            group_items.add(iitem.itemLocalId());
          }
        }
        group.addItems(group_items);
        debug() << "Incremental synchronization for the group <" << group.name() << ">"
                << " old=" << current_size << " new=" << group.size()
                << " added=" << group_items.size() << " removed=" << removed_size;
      }
      else {
        // We use the direct assignment mode for groups
        group_items.clear();
        ENUMERATE_ITEM(iitem,all_items){
          if ((m_var[iitem] & current_mask) != 0) {
            group_items.add(iitem.itemLocalId());
          }
        }
        // Preserving previous behavior: uses createGroup and not findGroup + setItems
        group.setItems(group_items);
        debug() << "Direct synchronization for the group <" << group.name() << ">"
                << " old=" << current_size << " new=" << group.size();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemGroupsSynchronize::
checkSynchronize()
{
  // TODO: check that all sub-domains have the same groups.
  Integer nb_diff = 0;
  for( const ItemGroup& group : m_item_family->groups() ){
    if (group.isOwn()) continue;
    m_var.fill(0);
    ENUMERATE_ITEM(i_item,group){
      m_var[*i_item] = 1;
    }
    Integer diff = m_var.checkIfSync(10);
    if (diff!=0){
      error() << "Group is not in sync (name=" << group.name()
              << ", nb_diff=" << diff << ")";
    }
    nb_diff += diff;
  }
  return nb_diff;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
