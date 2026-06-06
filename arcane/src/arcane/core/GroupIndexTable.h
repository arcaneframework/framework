// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GroupIndexTable.h                                           (C) 2000-2024 */
/*                                                                           */
/* Hash table between an item and its position in the table.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_GROUPINDEXTABLE_H
#define ARCANE_CORE_GROUPINDEXTABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ItemGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT GroupIndexTableView
{
  friend class GroupIndexTable;
  typedef Int32 KeyTypeValue;
  typedef Int32 ValueType;
  typedef HashTraitsT<KeyTypeValue> KeyTraitsType;
  typedef KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;

 public:

  ARCCORE_HOST_DEVICE ValueType operator[](Int32 i) const { return _lookup(i); }
  ARCCORE_HOST_DEVICE Int32 size() const { return m_key_buffer_span.size(); }

 private:

  SmallSpan<const KeyTypeValue> m_key_buffer_span;
  SmallSpan<const Int32> m_next_buffer_span;
  SmallSpan<const Int32> m_buckets_span;
  Int32 m_nb_bucket = 0;

 private:

  //! Search for a key in the entire table
  ARCCORE_HOST_DEVICE Int32 _lookup(KeyTypeConstRef id) const
  {
    return _lookupBucket(_hash(id), id);
  }
  ARCCORE_HOST_DEVICE Int32 _hash(KeyTypeConstRef id) const
  {
    return static_cast<Int32>(KeyTraitsType::hashFunction(id) % m_nb_bucket);
  }
  ARCCORE_HOST_DEVICE Integer _lookupBucket(Int32 bucket, KeyTypeConstRef id) const
  {
    for (Integer i = m_buckets_span[bucket]; i >= 0; i = m_next_buffer_span[i]) {
      if (m_key_buffer_span[i] == id)
        return i;
    }
    return -1;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class of a hash table between group items
 * and their positions in the table.
 *
 * This table is used for partial variables: the position of
 * an entity's data is not its localId() but its position in
 * the group enumerator (i.e.: in the table).         
 */
class ARCANE_CORE_EXPORT GroupIndexTable
: public HashTableBase
{
 public:

  typedef Int32 KeyTypeValue;
  typedef Int32 ValueType;
  typedef HashTraitsT<KeyTypeValue> KeyTraitsType;
  typedef KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;

 public:

  explicit GroupIndexTable(ItemGroupImpl* group_impl);

 public:

  void update();

  void clear();

  void compact(const Int32ConstArrayView* info);

  ValueType operator[](Int32 i) const { return _lookup(i); }

  KeyTypeValue keyLocalId(Int32 i) const { return m_key_buffer[i]; }

  Int32 size() const { return m_key_buffer.size(); }

  GroupIndexTableView view() const
  {
    ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
    ARCANE_ASSERT((_checkIntegrity(false)), ("GroupIndexTable integrity failed"));
    return m_view;
  }

 private:

  /*!
   * \brief Hashing function.
   *
   * Uses the Arcane hash function even if some
   * collisions are observed with small values.
   */
  Int32 _hash(KeyTypeConstRef id) const
  {
    ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
    return m_view._hash(id);
  }
  //! \a true if a value with key \a id is present
  bool _hasKey(KeyTypeConstRef id) const;

  //! Search for a key in a bucket
  Int32 _lookupBucket(Int32 bucket, KeyTypeConstRef id) const
  {
    ARCANE_ASSERT((_initialized()), ("GroupIndexTable not initialized"));
    return m_view._lookupBucket(bucket, id);
  }

  //! Search for a key in the entire table
  Int32 _lookup(KeyTypeConstRef id) const
  {
    ARCANE_ASSERT((_checkIntegrity(false)), ("GroupIndexTable integrity failed"));
    return _lookupBucket(_hash(id), id);
  }

  //! Tests the initialization of the object
  bool _initialized() const;

  //! Tests the integrity of the table relative to its group
  bool _checkIntegrity(bool full = true) const;

 private:

  ItemGroupImpl* m_group_impl = nullptr;
  UniqueArray<KeyTypeValue> m_key_buffer; //! Associated keys table
  UniqueArray<Int32> m_next_buffer; //! Associated next index table
  UniqueArray<Int32> m_buckets; //! Bucket array
  bool m_disable_check_integrity = false;
  GroupIndexTableView m_view;

 private:

  void _updateSpan();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
