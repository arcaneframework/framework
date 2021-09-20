// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilySerializer.cc                                     (C) 2000-2018 */
/*                                                                           */
/* Unique Serializer valid for any item family.                              */
/* Requires the use of the family graph: ItemFamilyNetwork                   */
/*---------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IItemFamily.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ConnectivityItemVector.h"

#include "arcane/mesh/ItemFamilySerializer.h"
#include "arcane/mesh/ItemFamilyNetwork.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilySerializer::
serializeItems(ISerializer* buf, Int32ConstArrayView local_ids)
{
  m_family->traceMng()->debug(Trace::High) << "_serializeItems : "
                                           << m_family->mesh()->name() << " "
                                           << m_family->name() << " "
                                           << m_family->parentFamilyDepth();

  Int32UniqueArray created_item_lids(local_ids.size()); // todo ne pas donner les lids car ils ne sont pas utilisés ensuite
  // Serialize items along with their dependencies
  ItemData item_dependencies_data(local_ids.size(),0,created_item_lids, m_family, m_family_modifier, m_family->parallelMng()->commRank());
  _fillItemDependenciesData(item_dependencies_data,local_ids);
  item_dependencies_data.serialize(buf); // ItemData handles the state of the ISerializer
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilySerializer::
deserializeItems(ISerializer* buf,Int32Array* local_ids)
{
  _deserializeItemsOrRelations(buf, local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilySerializer::
_deserializeItemsOrRelations(ISerializer *buf, Int32Array *local_ids)
{
  // Deserialize items (if m_deserialize_items) or items relations
  ItemData item_data; // it may be dependencies or relations
  if (local_ids) {
    item_data.deserialize(buf, m_family->mesh(), *local_ids);
  }
  else {
    item_data.deserialize(buf, m_family->mesh());
  }

  // Add items (or append relations to existing items)
  m_mesh_builder->addFamilyItems(item_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilySerializer::
serializeItemRelations(Arcane::ISerializer *buf, Arcane::Int32ConstArrayView local_ids)
{
  m_family->traceMng()->debug(Trace::High) << "_serializeItems relations : "
                                           << m_family->mesh()->name() << " "
                                           << m_family->name() << " "
                                           << m_family->parentFamilyDepth();

  Int32UniqueArray created_item_lids(local_ids.size()); // todo ne pas donner les lids car ils ne sont pas utilisés ensuite
  // Serialize items along with their relations
  ItemData item_relations_data(local_ids.size(),0,created_item_lids, m_family, m_family_modifier, m_family->parallelMng()->commRank());
  _fillItemRelationsData(item_relations_data,local_ids);
  item_relations_data.serialize(buf); // ItemData handles the state of the ISerializer
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilySerializer::
deserializeItemRelations(ISerializer* buf,Int32Array* local_ids)
{
  _deserializeItemsOrRelations(buf, local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilySerializer::
_fillItemDependenciesData(ItemData& item_data, Int32ConstArrayView local_ids)
{
  Int64Array&    item_infos  = item_data.itemInfos();
  Int32ArrayView item_owners = item_data.itemOwners();
  // Reserve size
  const Integer nb_item = local_ids.size();
  item_infos.reserve(1+nb_item*32); // Size evaluation for hexa cell (the more data to store) : 1_family_info + nb_item *(2_info_per_family + 6 (faces) + 12 (edges) + 8 (vertices) connected elements) = 1 + nb_item *(6 + 6 + 12 +8)
  // Fill item data (cf ItemData.h)
  IItemFamilyNetwork* family_network = m_family->mesh()->itemFamilyNetwork();
  auto out_connectivities = family_network->getChildDependencies(m_family); // Only dependencies are needed to create item. Relations are treated separately
  item_infos.add(out_connectivities.size());
  ENUMERATE_ITEM(item, m_family->view(local_ids))
  {
    item_infos.add(item->type());
    item_infos.add(item->uniqueId().asInt64());
    item_owners[item.index()] = item->owner();
    for (auto out_connectivity : out_connectivities)
    {
      if(!out_connectivity->isEmpty())
      {
        item_infos.add(out_connectivity->targetFamily()->itemKind());
        item_infos.add(out_connectivity->nbConnectedItem(ItemLocalId(item)));
        ConnectivityItemVector connectivity_accessor(out_connectivity);
        ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(item))) {
          item_infos.add(connected_item->uniqueId().asInt64());
        }
      }
      else
      {
        item_infos.add(out_connectivity->sourceFamily()->itemKind());
        item_infos.add(0) ;
      }
    }
  }
}

void ItemFamilySerializer::
_fillItemRelationsData(ItemData& item_data, Int32ConstArrayView local_ids)
{
  // Fill item relations, ie upward connectivities and extra connectivities (eg dof)
  Int64Array&    item_infos  = item_data.itemInfos();
  Int32ArrayView item_owners = item_data.itemOwners();
  // Reserve size
  const Integer nb_item = local_ids.size();
  item_infos.reserve(1+nb_item*32); // Size evaluation for hexa cell (the more data to store) : 1_family_info + nb_item *(2_info_per_family + 6 (faces) + 12 (edges) + 8 (vertices) connected elements) = 1 + nb_item *(6 + 6 + 12 +8)
  // Fill item data (cf ItemData.h)
  IItemFamilyNetwork* family_network = m_family->mesh()->itemFamilyNetwork();
  auto out_connectivities = family_network->getChildRelations(m_family); // Only relations are taken
  item_infos.add(out_connectivities.size());
  ENUMERATE_ITEM(item, m_family->view(local_ids)){
    item_infos.add(item->type());
    item_infos.add(item->uniqueId().asInt64());
    item_owners[item.index()] = item->owner();
    for (auto out_connectivity : out_connectivities)
    {
      item_infos.add(out_connectivity->targetFamily()->itemKind());
      item_infos.add(out_connectivity->nbConnectedItem(ItemLocalId(item)));
      ConnectivityItemVector connectivity_accessor(out_connectivity);
      ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(item)))
      {
        item_infos.add(connected_item->uniqueId().asInt64());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemFamilySerializer::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
