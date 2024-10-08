﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalMap.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Tableau associatif de ItemInternal.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemInternalMap.h"

#include "arcane/utils/Iterator.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/core/Item.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

#define ENUMERATE_ITEM_INTERNAL_MAP_DATA2(iter, item_list) \
  for (auto __i__##iter : item_list.buckets()) \
    for (auto* iter = __i__##iter; iter; iter = iter->next())

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalMap::
ItemInternalMap()
: m_impl(5000, false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternalMap::
notifyUniqueIdsChanged()
{
  if (arcaneIsCheck()) {
    // Vérifie qu'on n'a pas deux fois la même clé.
    std::unordered_set<Int64> uids;
    this->eachItem([&](Item item) {
      Int64 uid = item.uniqueId().asInt64();
      if (uids.find(uid) != uids.end())
        ARCANE_FATAL("Duplicated uniqueId '{0}'", uid);
      uids.insert(uid);
    });
  }

  if constexpr (UseNewImpl) {
    Int64 nb_item = m_new_impl.size();
    UniqueArray<ItemInternal*> items(nb_item);
    Int64 index = 0;
    for (auto& x : m_new_impl) {
      items[index] = x.second;
      ++index;
    }
    m_new_impl.clear();
    for (index = 0; index < nb_item; ++index) {
      ItemInternal* item = items[index];
      m_new_impl.insert(std::make_pair(item->uniqueId(), item));
    }
  }
  else {
    ENUMERATE_ITEM_INTERNAL_MAP_DATA2 (nbid, m_impl) {
      nbid->setKey(nbid->value()->uniqueId().asInt64());
    }

    m_impl.rehash();
  }

  checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternalMap::
_changeLocalIds(ArrayView<ItemInternal*> items_internal,
                ConstArrayView<Int32> old_to_new_local_ids)
{
  checkValid();

  if constexpr (UseNewImpl) {
    for (auto& iter : m_new_impl) {
      ItemInternal* old_ii = iter.second;
      Int32 current_local_id = old_ii->localId();
      ItemInternal* new_ii = items_internal[old_to_new_local_ids[current_local_id]];
      iter.second = new_ii;
    }
  }
  else {
    ENUMERATE_ITEM_INTERNAL_MAP_DATA2 (nbid, m_impl) {
      ItemInternal* old_ii = nbid->value();
      Int32 current_local_id = old_ii->localId();
      ItemInternal* new_ii = items_internal[old_to_new_local_ids[current_local_id]];
      nbid->setValue(new_ii);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternalMap::
checkValid() const
{
  if (!arcaneIsCheck())
    return;

  if constexpr (UseNewImpl) {
    for (auto& x : m_new_impl) {
      if (x.first != x.second->uniqueId())
        ARCANE_FATAL("Incoherent uid key={0} item_internal={1}", x.first, x.second->uniqueId());
    }
  }
  else {
    ENUMERATE_ITEM_INTERNAL_MAP_DATA2(nbid, m_impl)
    {
      if (nbid->key() != nbid->value()->uniqueId())
        ARCANE_FATAL("Incoherent uid key={0} item_internal={1}", nbid->key(), nbid->value()->uniqueId());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternalMap::
_throwNotFound(Int64 key) const
{
  ARCANE_FATAL("ERROR: can not find key={0}", key);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternalMap::
_throwNotSupported(const char* func_name) const
{
  ARCANE_THROW(NotSupportedException, func_name);
}

void ItemInternalMap::
_checkValid(Int64 uid, ItemInternal* v) const
{
  if (v->uniqueId() != uid)
    ARCANE_FATAL("Bad found uniqueId found={0} expected={1}", v->uniqueId(), uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
