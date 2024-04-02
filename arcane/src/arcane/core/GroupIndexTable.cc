// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GroupIndexTable.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Table de hachage entre un item et sa position dans la table.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/ItemGroupImpl.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/GroupIndexTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GroupIndexTable::
GroupIndexTable(ItemGroupImpl* group_impl)
: HashTableBase(0, false)
, m_group_impl(group_impl)
{
  ARCANE_ASSERT((m_group_impl), ("ItemGroupImpl pointer null"));
#ifdef ARCANE_ASSERT
  m_disable_check_integrity = platform::getEnvironmentVariable("ARCANE_ENABLE_GROUPINDEXTABLE_CHECKINTEGRITY").null();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GroupIndexTable::
update()
{
  ItemGroup group(m_group_impl); // will update group if necessary

  const Integer group_size = group.size();
  m_nb_bucket = this->nearestPrimeNumber(2 * group_size);
  m_buckets.resize(m_nb_bucket);
  m_buckets.fill(-1);
  m_key_buffer.resize(group_size);
  m_next_buffer.resize(group_size);

  ENUMERATE_ITEM (item, group) {
    const Integer index = item.index();
    const KeyTypeConstRef key = item->localId();
    const Integer bucket = _hash(key);
    ARCANE_ASSERT((_lookupBucket(bucket, key) < 0), ("Already assigned key"));
    m_key_buffer[index] = key;
    m_next_buffer[index] = m_buckets[bucket];
    m_buckets[bucket] = index;
  }

  ARCANE_ASSERT((_checkIntegrity()), ("GroupIndexTable integrity failed"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GroupIndexTable::
compact(const Int32ConstArrayView* infos)
{
  ARCANE_UNUSED(infos);
  // Avec cette version, on suppose que l'ordre relatif des ids n'a pas changé
  // La taille du groupe n'a pas changé mais on réordonne les données
  ARCANE_ASSERT((m_group_impl->size() == m_key_buffer.size()), ("Inconsistent sizes"));

#ifdef NDEBUG
  update();
#else /* NDEBUG */
  // identique à update() mais peut faire quelque contrôle si infos!=NULL
  m_buckets.fill(-1);
  ItemGroup group(m_group_impl);
  ENUMERATE_ITEM (iitem, group) {
    const KeyTypeConstRef key = iitem.localId();
    const Integer i = iitem.index();
    const KeyTypeConstRef old_key = m_key_buffer[iitem.index()];
    const Integer bucket = _hash(key);
    ARCANE_ASSERT((_lookupBucket(bucket, key) < 0), ("Already assigned key"));
    ARCANE_ASSERT((infos == NULL || (*infos)[old_key] == key), ("Inconsistent reorder translation %d vs %d vs %d", (*infos)[old_key], key, old_key));
    m_key_buffer[i] = key;
    m_next_buffer[i] = m_buckets[bucket];
    m_buckets[bucket] = i;
  }
  ARCANE_ASSERT((_checkIntegrity()), ("GroupIndexTable integrity failed"));
#endif /* NDEBUG */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer GroupIndexTable::
_hash(KeyTypeConstRef id) const
{
  ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
  return (Integer)(KeyTraitsType::hashFunction(id) % m_nb_bucket);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GroupIndexTable::
_hasKey(KeyTypeConstRef id) const
{
  const Integer hf = _hash(id);
  for (Integer i = m_buckets[hf]; i >= 0; i = m_next_buffer[i]) {
    if (m_key_buffer[i] == id)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer GroupIndexTable::
_lookupBucket(Integer bucket, KeyTypeConstRef id) const
{
  ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
  for (Integer i = m_buckets[bucket]; i >= 0; i = m_next_buffer[i]) {
    if (m_key_buffer[i] == id)
      return i;
  }
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer GroupIndexTable::
_lookup(KeyTypeConstRef id) const
{
  ARCANE_ASSERT((_checkIntegrity(false)), ("GroupIndexTable integrity failed"));
  return _lookupBucket(_hash(id), id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GroupIndexTable::
_initialized() const
{
  return !m_buckets.empty();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GroupIndexTable::
_checkIntegrity(bool full) const
{
  bool integrity = true;
  if (m_group_impl->size() != m_key_buffer.size()) {
    StringBuilder b;
    b += "GroupIndexTable BAD DATA size : group size=";
    b += m_group_impl->size();
    b += " vs data size=";
    b += m_key_buffer.size();
    throw FatalErrorException(b.toString());
    integrity = false;
  }
#ifdef ARCANE_ASSERT
  if (!full && m_disable_check_integrity)
    return integrity;
#endif
  ItemGroup group(m_group_impl);
  ENUMERATE_ITEM (item, group) {
    const Integer hd = item.index();
    if (m_key_buffer[hd] != item.localId() || !_hasKey(item.localId())) {
      StringBuilder b;
      b += "GroupIndexTable BAD DATA at ";
      b += item.index();
      b += " :  lid=";
      b += item.localId();
      b += m_key_buffer[hd];
      b += " chk:";
      b += _hasKey(item.localId());
      throw FatalErrorException(b.toString());
      integrity = false;
    }
  }
  return integrity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
