// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*------------------------------------------------------------------------------*/
/* ReplicatedGhostDoFBuilder.cc                                   (C) 2000-2017 */
/*                                                                              */
/* Implémentation d'une politique de création de fantômes pour une connectivité */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "GhostLayerFromConnectivityComputer.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableSynchronizer.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::GhostLayerFromConnectivityComputer::
GhostLayerFromConnectivityComputer(IItemConnectivity* item_to_dofs)
  : m_connectivity(item_to_dofs)
  , m_trace_mng(m_connectivity->sourceFamily()->traceMng())
{
  _computeSharedItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
Arcane::GhostLayerFromConnectivityComputer::
_computeSharedItems()
{
  m_shared_items.clear();
  m_shared_items_connected_items.clear();
  // Politique : les shared items de la famille To sont ceux qui sont connectés aux shared de la famille from
  // Attention ici le tableau shared items contient des items de la famille To family
  Arcane::IItemFamily* item_family = m_connectivity->sourceFamily();
  Arcane::IVariableSynchronizer* synchronizer = item_family->allItemsSynchronizer();
  Arcane::Int32ConstArrayView ranks = synchronizer->communicatingRanks();
  m_shared_items.resize(synchronizer->parallelMng()->commSize());
  m_shared_items_connected_items.resize(synchronizer->parallelMng()->commSize());

  for (Arcane::Integer i = 0; i < ranks.size(); ++i){
    Arcane::ItemVectorView from_family_shared_items_view = item_family->view(synchronizer->sharedItems(i));
    Arcane::ConnectivityItemVector con_items(m_connectivity);
    Arcane::Int32Array& shared_items = m_shared_items[ranks[i]];
    Arcane::Int32Array& shared_items_connected_items = m_shared_items_connected_items[ranks[i]];

    ENUMERATE_ITEM(from_family_item,from_family_shared_items_view){
      ENUMERATE_ITEM(to_family_item,con_items.connectedItems(from_family_item)){
        shared_items.add(to_family_item.localId());
        shared_items_connected_items.add(from_family_item.localId());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
Arcane::GhostLayerFromConnectivityComputer::
updateConnectivity(Int32ConstArrayView ghost_items, Int64ConstArrayView ghost_items_connected_items)
{
  // Very simple case : ghost_items are connected to the items to witch they where connected in their owner
  // => ghost_items_connected items are present as ghost item in the current sub-domain
  Arcane::Int32SharedArray ghost_items_connected_items_lids(ghost_items_connected_items.size());
  bool do_fatal = true;
  m_connectivity->sourceFamily()->itemsUniqueIdToLocalId(ghost_items_connected_items_lids,ghost_items_connected_items,do_fatal);
  m_connectivity->updateConnectivity(ghost_items_connected_items_lids,ghost_items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int32ConstArrayView
Arcane::GhostLayerFromConnectivityComputer::
communicatingRanks()
{
  return m_connectivity->sourceFamily()->allItemsSynchronizer()->communicatingRanks();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int32ConstArrayView
Arcane::GhostLayerFromConnectivityComputer::
sharedItems(const Integer rank, const String& family_name)
{
  ARCANE_ASSERT((family_name == m_connectivity->targetFamily()->name()),
                (String::format("Error : asking shared item for the family {0} that is not the ToFamily ({1}) of the connectivity",family_name,m_connectivity->targetFamily()->name()).localstr()))
  _computeSharedItems();
  return m_shared_items[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int32ConstArrayView
Arcane::GhostLayerFromConnectivityComputer::
sharedItemsConnectedItems(const Integer rank, const String& family_name)
{
  ARCANE_ASSERT((family_name == m_connectivity->sourceFamily()->name()),
                (String::format("Error : asking shared item connected items for the family {0} that is not the FromFamily ({1}) of the connectivity",family_name,m_connectivity->sourceFamily()->name()).localstr()))
  return m_shared_items_connected_items[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
