// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSynchronize.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Synchronisations des groupes.                                             */
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

  // Construction de la liste des groupes
  UniqueArray<ItemGroup> groups;
  CommonItemGroupFilterer group_filterer(m_item_family);
  for( ItemGroup group : m_groups ){
    ARCANE_ASSERT((!group.null()),("Null group in ItemGroupsSynchronize"));
    groups.add(group);
    group_filterer.addGroupToFilter(group);
  }
  // TODO: regarder s'il ne faudrait pas le faire à chaque fois
  // Comme cela on serait certains que les groupes sont bien triés
  if (arcaneIsCheck())
    group_filterer.applyFiltering();

  Int32UniqueArray group_items; // Tableau stockant les listes d'items en évolution
  group_items.reserve(m_item_family->maxLocalId());

  const Integer max_aggregate_size = 
    sizeof(IntAggregator)*8-1; // Pour rester dans l'espace des entiers positifs
  const Integer aggregate_count = 
    (groups.size() / max_aggregate_size) +
    ((groups.size() % max_aggregate_size)?1:0);

  for(Integer i_aggregate=0; i_aggregate < aggregate_count; ++i_aggregate) {
    const Integer first_group = i_aggregate * max_aggregate_size;
    const Integer current_aggregate_size = math::min(max_aggregate_size, 
                                                     groups.size()-first_group);
    m_var.fill(0); // Initialisation du tableau parallèle
    for(Integer i_group=0;i_group<current_aggregate_size;++i_group) {
      const IntAggregator current_mask = static_cast<IntAggregator>(1)<<i_group;
      ItemGroup group = groups[first_group+i_group];
      ENUMERATE_ITEM(iitem,group){
        m_var[iitem] |= current_mask;
      }
    }

    // Partage des infos de contenu des groupes sur l'aggrégat
    m_var.synchronize();

    for(Integer i_group=0;i_group<current_aggregate_size;++i_group) {
      const IntAggregator current_mask = static_cast<IntAggregator>(1)<<i_group;
      ItemGroup group = groups[first_group+i_group];
      const Integer current_size = group.size();
      if (group.internal()->hasInfoObserver()) {
        // Passage en mode incrémental
        // Recherche des items disparus
        group_items.clear();
        ENUMERATE_ITEM(iitem, group) {
          if ((m_var[iitem] & current_mask) == 0) {
            group_items.add(iitem.itemLocalId());
          } else {
            m_var[iitem] &= ~current_mask; // marque comme déjà existant dans le groupe
          }
        }
        Integer removed_size = group_items.size();
        group.removeItems(group_items);
        // Recherche des nouveaux items
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
        // On utilise le mode d'affectation direct des groupes
        group_items.clear();
        ENUMERATE_ITEM(iitem,all_items){
          if ((m_var[iitem] & current_mask) != 0) {
            group_items.add(iitem.itemLocalId());
          }
        }
        // Préservation du précédent comportement : utilise createGroup et non findGroup + setItems
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
  // TODO: vérifier que tous les sous-domaines ont les mêmes groupes.
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
