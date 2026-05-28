// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSerializer2.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Serialization of entity groups.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/Item.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ItemFamilySerializeArgs.h"

#include "arcane/mesh/ItemGroupsSerializer2.h"

#include <algorithm>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupsSerializer2::
ItemGroupsSerializer2(IItemFamily* item_family, IParallelExchanger* exchanger)
: TraceAccessor(item_family->traceMng())
, m_exchanger(exchanger)
, m_mesh(item_family->mesh())
, m_item_family(item_family)
{
  // The list of groups in itemFamily()->groups() is not necessarily the
  // same for all subdomains. We therefore use a std::map to sort them
  // alphabetically so that everything in m_groups_to_exchange is in the same order.
  std::map<String, ItemGroup> group_set;

  for (ItemGroupCollection::Enumerator i_group(itemFamily()->groups()); ++i_group;) {
    ItemGroup group = *i_group;
    if (group.internal()->needSynchronization()) {
      group_set.insert(std::make_pair(group.name(), group));
    }
  }

  for (const auto& iter : group_set) {
    const ItemGroup& group = iter.second;
    m_groups_to_exchange.add(group);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupsSerializer2::
~ItemGroupsSerializer2()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupsSerializer2::
serialize(const ItemFamilySerializeArgs& args)
{
  ISerializer* sbuf = args.serializer();
  Int32 rank = args.rank();
  // NOTE: for the moment args.localIds() is not used.
  switch (sbuf->mode()) {
  case ISerializer::ModeReserve:
    sbuf->reserveInt64(1);
    sbuf->reserveSpan(eBasicDataType::Int64, m_items_to_send[rank].size());
    break;
  case ISerializer::ModePut:
    sbuf->putInt64(m_items_to_send[rank].size());
    sbuf->putSpan(m_items_to_send[rank]);
    break;
  case ISerializer::ModeGet:
    ARCANE_FATAL("Do no call this method for deserialization. Use method get()");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupsSerializer2::
get(ISerializer* sbuf, Int64Array& items_in_groups_uid)
{
  // Retrieves the list of uniqueId() of the group entities
  Int64 nb_item_in_groups = sbuf->getInt64();
  items_in_groups_uid.resize(nb_item_in_groups);
  sbuf->getSpan(items_in_groups_uid);

  info(4) << "Receiving groups family=" << itemFamily()->fullName()
          << " n=" << nb_item_in_groups << " entities";

  Integer local_index = 0;
  UniqueArray<Int32> items_in_group_local_id;
  [[maybe_unused]] Int32 group_index = 0;
  for (ItemGroupList::Enumerator i_group(m_groups_to_exchange); ++i_group; ++group_index) {
    ItemGroup group = *i_group;
    // The first element of the array contains the number of items
    Integer nb_item_in_group = CheckedConvert::toInteger(items_in_groups_uid[local_index]);
    ++local_index;
    if (nb_item_in_group != 0) {
      Int64ArrayView items_in_group(nb_item_in_group, &items_in_groups_uid[local_index]);
#if 0
      info() << "Unserialize group " << group.name() << " index=" << group_index << " nb item " << nb_item_in_group;
      for( Integer z=0; z<nb_item_in_group; ++z ){
        info() << "UID = " << items_in_group[z];
      }
#endif
      items_in_group_local_id.resize(nb_item_in_group);
      itemFamily()->itemsUniqueIdToLocalId(items_in_group_local_id, items_in_group);
      group.addItems(items_in_group_local_id);
    }
    local_index += nb_item_in_group;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupsSerializer2::
prepareData(ConstArrayView<SharedArray<Int32>> items_exchange)
{
  IParallelMng* pm = m_mesh->parallelMng();

  {
    // Checks that everyone has the same number of groups
    // TODO: it should also check that they are the same.
    // TODO: this test can be removed if we are sure everything is OK.
    Integer nb_group = m_groups_to_exchange.count();
    Integer total_nb_group = pm->reduce(Parallel::ReduceMax, nb_group);
    if (nb_group != total_nb_group) {
      for (ItemGroupList::Enumerator i_group(m_groups_to_exchange); ++i_group;) {
        ItemGroup group = *i_group;
        info() << "Group: " << group.name();
      }
      fatal() << "Number of groups different between subdomains"
              << " family=" << itemFamily()->fullName()
              << " current=" << nb_group
              << " max=" << total_nb_group;
    }
  }

  Integer nb_sub_domain = pm->commSize();
  // Contains for each processor the list of uniqueId() of the group entities
  // to be transferred
  m_items_to_send.resize(nb_sub_domain);
  UniqueArray<Integer> first_items_to_send(m_items_to_send.size());
  first_items_to_send.fill(0);

  UniqueArray<bool> items_in_exchange(itemFamily()->maxLocalId());

  Int32ConstArrayView send_sub_domains(m_exchanger->senderRanks());
  for (Integer i = 0, is = send_sub_domains.size(); i < is; ++i) {
    Integer dest_sub_domain = send_sub_domains[i];
    items_in_exchange.fill(false);
    Int32ConstArrayView items_exchange_lid(items_exchange[dest_sub_domain]);
    for (Integer z = 0, zs = items_exchange_lid.size(); z < zs; ++z)
      items_in_exchange[items_exchange_lid[z]] = true;

    for (ItemGroupList::Enumerator i_group(m_groups_to_exchange); ++i_group;) {
      ItemGroup group = *i_group;
      info(4) << "Serialize group " << group.name();
      first_items_to_send[dest_sub_domain] = m_items_to_send[dest_sub_domain].size();
      m_items_to_send[dest_sub_domain].add(NULL_ITEM_ID);

      //_prepareData(group,m_items_to_send);

      ENUMERATE_ITEM (iitem, group) {
        if (items_in_exchange[iitem.itemLocalId()])
          m_items_to_send[dest_sub_domain].add((*iitem).uniqueId().asInt64());
      }

      Integer first_item = first_items_to_send[dest_sub_domain];
      Integer last_item = m_items_to_send[dest_sub_domain].size() - 1;
      info(5) << "Serialize for subdomain " << dest_sub_domain
              << " first " << first_item << " last " << last_item;
      m_items_to_send[dest_sub_domain][first_item] = last_item - first_item;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
