// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTable.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Hash Table.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHTABLE_H
#define ARCANE_UTILS_HASHTABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MultiBuffer.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/HashFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for a simple hash table for entities

 \todo Add iterators for this collection and derived classes
 */
class ARCANE_UTILS_EXPORT HashTableBase
{
 public:

  /*!
   * \brief Creates a table of size \a table_size
   *
   * If \a use_prime is true, uses the nearestPrimeNumber() function
   * to have a size that is a prime number.
   */
  HashTableBase(Integer table_size, bool use_prime)
  : m_count(0)
  , m_nb_bucket(use_prime ? nearestPrimeNumber(table_size) : table_size)
  {
  }
  virtual ~HashTableBase() {}

 public:

  /*!
   * \brief Returns the nearest prime number greater than \a n.
   * The nearest prime number greater than \a n is returned using a
   * pre-determined prime number table.
   */
  Integer nearestPrimeNumber(Integer n);

 public:

  //! Number of elements in the table
  Integer count() const
  {
    return m_count;
  }

 protected:

  void _throwNotFound ARCANE_NORETURN() const;

 protected:

  Integer m_count; //!< Number of elements
  Integer m_nb_bucket; //!< Number of buckets
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Base class for a hash table for associative arrays
 
 This table allows storing a value based on a key. The
 value type is managed by the derived class HashTableMapT.
 
 The hash table is managed as an array whose number
 of elements is given by the table size (m_nb_bucket).
 The elements are then stored in a linked list.

 This table only allows adding values.

 For performance reasons, it is preferable that the table size
 (buckets) is a prime number.
 */
template <typename KeyType, typename TraitsType>
class HashTableBaseT
: public HashTableBase
{
 public:

  typedef typename TraitsType::KeyTypeConstRef KeyTypeConstRef;
  typedef typename TraitsType::KeyTypeValue KeyTypeValue;

 public:

  struct HashData
  {
   public:

    friend class HashTableBaseT<KeyType, TraitsType>;

   public:

    HashData()
    : m_key(KeyType())
    , m_next(0)
    {}

   public:

    /*!
     * \brief Changes the key value.
     *
     * After changing the value of one or more keys, a rehash must be performed().
     */
    void changeKey(const KeyType& new_key)
    {
      m_key = new_key;
    }

   protected:

    KeyTypeValue m_key; //!< Search key
    HashData* m_next; //! Next element in the hash table
  };

 public:

  /*!
   * \brief Creates a table of size \a table_size
   *
   * If \a use_prime is true, uses the nearestPrimeNumber() function
   * to have a size that is a prime number.
  */
  HashTableBaseT(Integer table_size, bool use_prime)
  : HashTableBase(table_size, use_prime)
  , m_buckets(m_nb_bucket)
  {
    m_buckets.fill(0);
  }

 public:

  //! \a true if a value with the key \a id is present
  bool hasKey(KeyTypeConstRef id) const
  {
    KeyType hf = _hash(id);
    for (HashData* i = m_buckets[hf]; i; i = i->m_next) {
      if (i->m_key == id)
        return true;
    }
    return false;
  }

  //! Clears all elements from the table
  void clear()
  {
    m_buckets.fill(0);
    m_count = 0;
  }

  //! Resizes the hash table
  void resize(Integer new_table_size, bool use_prime = false)
  {
    m_nb_bucket = new_table_size;
    if (new_table_size == 0) {
      clear();
      return;
    }
    if (use_prime)
      new_table_size = nearestPrimeNumber(new_table_size);
    //todo: remove the allocation of this array
    UniqueArray<HashData*> old_buckets(m_buckets.clone());
    m_buckets.resize(new_table_size);
    m_buckets.fill(0);
    for (Integer z = 0, zs = old_buckets.size(); z < zs; ++z) {
      for (HashData* i = old_buckets[z]; i; i = i->m_next) {
        _baseAdd(_hash(i->m_key), i->m_key, i);
      }
    }
  }

  //! Repositions data after key value change
  void rehash()
  {
    //todo: remove the allocation of this array
    UniqueArray<HashData*> old_buckets(m_buckets.clone());
    m_buckets.fill(0);

    for (Integer z = 0, zs = old_buckets.size(); z < zs; ++z) {
      for (HashData* i = old_buckets[z]; i;) {
        HashData* current = i;
        i = i->m_next; // Must be done here, because i->m_next changes with _baseAdd
        _baseAdd(_hash(current->m_key), current->m_key, current);
      }
    }
  }

 protected:

  inline Integer _hash(KeyTypeConstRef id) const
  {
    return TraitsType::hashFunction(id) % m_nb_bucket;
  }
  inline HashData* _baseLookupBucket(Integer bucket, KeyTypeConstRef id) const
  {
    for (HashData* i = m_buckets[bucket]; i; i = i->m_next) {
      if (i->m_key == id)
        return i;
    }
    return 0;
  }
  inline HashData* _baseRemoveBucket(Integer bucket, KeyTypeConstRef id)
  {
    HashData* i = m_buckets[bucket];
    if (i) {
      if (i->m_key == id) {
        m_buckets[bucket] = i->m_next;
        --m_count;
        return i;
      }
      for (; i->m_next; i = i->m_next) {
        if (i->m_next->m_key == id) {
          HashData* r = i->m_next;
          i->m_next = i->m_next->m_next;
          --m_count;
          return r;
        }
      }
    }
    _throwNotFound();
  }
  inline HashData* _baseLookup(KeyTypeConstRef id) const
  {
    return _baseLookupBucket(_hash(id), id);
  }
  inline HashData* _baseRemove(KeyTypeConstRef id)
  {
    return _baseRemoveBucket(_hash(id), id);
  }
  inline void _baseAdd(Integer bucket, KeyTypeConstRef id, HashData* hd)
  {
    HashData* buck = m_buckets[bucket];
    hd->m_key = id;
    hd->m_next = buck;
    m_buckets[bucket] = hd;
    ++m_count;
  }

 protected:

  UniqueArray<HashData*> m_buckets; //! Array of buckets
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
