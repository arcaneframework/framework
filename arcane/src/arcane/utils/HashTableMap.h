// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTableMap.h                                              (C) 2000-2024 */
/*                                                                           */
/* Associative array using a hash table.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHTABLEMAP_H
#define ARCANE_UTILS_HASHTABLEMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename KeyType, typename ValueType>
class HashTableMapEnumeratorT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Hash table for associative arrays.
 
 This table allows storing a value based on a key. The key is of type \a KeyType and the value is \a ValueType.

 For now, this table only allows adding values.
 The memory associated with each entry in the array is managed by
 a MultiBufferT.

 It is possible to specify a hash function different from
 the default function by specifying the third template parameter \a KeyTraitsType.

 For performance reasons, it is preferable that the size
 of the table (buckets) is a prime number.
*/
template <typename KeyType, typename ValueType, typename KeyTraitsType = HashTraitsT<KeyType>>
class HashTableMapT
: public HashTableBase
{
 public:

  typedef typename KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;
  typedef typename KeyTraitsType::KeyTypeValue KeyTypeValue;
  typedef typename KeyTraitsType::Printable Printable;
  typedef typename KeyTraitsType::HashValueType HashValueType;
  typedef HashTableMapT<KeyType, ValueType, KeyTraitsType> ThatClass;
  typedef HashTableMapEnumeratorT<KeyType, ValueType> Enumerator;

 public:

  struct Data
  {
   public:

    Data()
    : m_key(KeyTypeValue())
    , m_value(ValueType())
    {}

   public:

    Data* next() { return m_next; }
    void setNext(Data* anext) { this->m_next = anext; }
    KeyTypeConstRef key() { return m_key; }
    const ValueType& value() const { return m_value; }
    ValueType& value() { return m_value; }
    //! Modifies the value of the instance.
    void setValue(const ValueType& avalue) { m_value = avalue; }
    /*!
     * \brief Changes the value of the key.
     *
     * After changing the value of one or more keys, a rehash() must be performed.
     */
    void setKey(const KeyType& new_key)
    {
      m_key = new_key;
    }

   public:

    KeyTypeValue m_key; //!< Search key
    ValueType m_value; //!< Element value
    Data* m_next = nullptr; //! Next element in the hash table
  };

 public:

  /*! \brief Creates a table of size \a table_size
   *
   If \a use_prime is true, it uses the nearestPrimeNumber() function
   to have a size that is a prime number.
  */
  HashTableMapT(Integer table_size, bool use_prime)
  : HashTableBase(table_size, use_prime)
  , m_first_free(0)
  , m_nb_collision(0)
  , m_nb_direct(0)
  , m_max_count(0)
  {
    m_buffer = new MultiBufferT<Data>(m_nb_bucket);
    m_buckets.resize(m_nb_bucket);
    m_buckets.fill(0);
    _computeMaxCount();
  }

  /*! \brief Creates a table of size \a table_size
   *
   If \a use_prime is true, it uses the nearestPrimeNumber() function
   to have a size that is a prime number.
  */
  HashTableMapT(Integer table_size, bool use_prime, Integer buffer_size)
  : HashTableBase(table_size, use_prime)
  , m_first_free(0)
  , m_nb_collision(0)
  , m_nb_direct(0)
  {
    m_buffer = new MultiBufferT<Data>(buffer_size);
    m_buckets.resize(m_nb_bucket);
    m_buckets.fill(0);
    _computeMaxCount();
  }

  ~HashTableMapT()
  {
    delete m_buffer;
  }

  //! Copy assignment operator
  ThatClass& operator=(const ThatClass& from)
  {
    if (&from == this)
      return *this;
    //cout << "** OPERATOR= this=" << this << '\n';
    Integer nb_bucket = from.m_nb_bucket;
    m_first_free = 0;
    // Resets the counter.
    m_count = 0;
    m_buckets.resize(nb_bucket);
    m_buckets.fill(0);
    _computeMaxCount();
    delete m_buffer;
    m_buffer = new MultiBufferT<Data>(nb_bucket);
    ConstArrayView<Data*> from_buckets(from.buckets());
    for (Integer i = 0; i < nb_bucket; ++i)
      for (Data* data = from_buckets[i]; data; data = data->next())
        _add(i, data->key(), data->value());
    this->m_nb_bucket = nb_bucket;
    return *this;
  }

 public:

  //! \a true if a value with key \a id is present
  bool hasKey(KeyTypeConstRef id)
  {
    Integer hf = _keyToBucket(id);
    for (Data* i = m_buckets[hf]; i; i = i->m_next) {
      if (i->key() == id)
        return true;
    }
    return false;
  }

  //! Deletes all elements from the table
  void clear()
  {
    m_buckets.fill(0);
    m_count = 0;
  }

  /*!
   * \brief Searches for the value corresponding to key \a id.
   *
   * \return the structure associated with key \a id (0 if none)
   */
  Data* lookup(KeyTypeConstRef id)
  {
    return _lookup(id);
  }

  /*!
   * \brief Searches for the value corresponding to key \a id.
   *
   * \return the structure associated with key \a id (0 if none)
   */
  const Data* lookup(KeyTypeConstRef id) const
  {
    return _lookup(id);
  }

  /*!
   * \brief Searches for the value corresponding to key \a id.
   *
   * An exception is generated if the value is not found.
   */
  ValueType& lookupValue(KeyTypeConstRef id)
  {
    Data* ht = _lookup(id);
    if (!ht) {
      this->_throwNotFound(id, Printable());
    }
    return ht->value();
  }

  /*!
   * \brief Searches for the value corresponding to key \a id.
   *
   * An exception is generated if the value is not found.
   */
  ValueType& operator[](KeyTypeConstRef id)
  {
    return lookupValue(id);
  }

  /*!
   * \brief Searches for the value corresponding to key \a id.
   *
   * An exception is generated if the value is not found.
   */
  const ValueType& lookupValue(KeyTypeConstRef id) const
  {
    const Data* ht = _lookup(id);
    if (!ht) {
      this->_throwNotFound(id, Printable());
    }
    return ht->m_value;
  }

  /*!
   * \brief Searches for the value corresponding to key \a id.
   *
   * An exception is generated if the value is not found.
   */
  const ValueType& operator[](KeyTypeConstRef id) const
  {
    return lookupValue(id);
  }

  /*!
   * \brief Adds the value \a value corresponding to key \a id
   *
   * If a value corresponding to \a id already exists, it is replaced.
   *
   * \retval true if the key is added
   * \retval false if the key already exists and is replaced
   */
  bool add(KeyTypeConstRef id, const ValueType& value)
  {
    Integer hf = _keyToBucket(id);
    Data* ht = _lookupBucket(hf, id);
    if (ht) {
      ht->m_value = value;
      return false;
    }
    _add(hf, id, value);
    _checkResize();
    return true;
  }

  /*!
   * \brief Removes the value associated with key \a id
   */
  void remove(KeyTypeConstRef id)
  {
    Integer hf = _keyToBucket(id);
    Data* ht = _removeBucket(hf, id);
    ht->setNext(m_first_free);
    m_first_free = ht;
  }

  /*!
   * \brief Searches for or adds the value corresponding to key \a id.
   * 
   * If key \a id is already in the table, returns a reference to this
   * value and sets \a is_add to \c false. Otherwise, adds key \a id
   * with value \a value and sets \a is_add to \c true.
   *
   * The returned structure is never null and can be kept because it
   * does not change address as long as this hash table instance exists
   */
  Data* lookupAdd(KeyTypeConstRef id, const ValueType& value, bool& is_add)
  {
    HashValueType hf = _applyHash(id);
    Data* ht = _lookupBucket(_hashValueToBucket(hf), id);
    if (ht) {
      is_add = false;
      return ht;
    }
    is_add = true;
    // Always perform the resize before returning the add
    // because it may invalidate the Data*
    _checkResize();
    ht = _add(_hashValueToBucket(hf), id, value);
    return ht;
  }

  /*!
   * \brief Searches for or adds the value corresponding to key \a id.
   *
   * If key \a id is already in the table, returns a reference to this
   * value and sets \a is_add to \c false. Otherwise, adds key \a id
   * with value \a ValueType() (which must exist).
   * 
   * The returned structure is never null and can be kept because it
   * does not change address as long as this hash table instance exists
   */
  Data* lookupAdd(KeyTypeConstRef id)
  {
    HashValueType hf = _applyHash(id);
    Data* ht = _lookupBucket(_hashValueToBucket(hf), id);
    if (!ht) {
      // Always perform the resize before returning the add
      // because it may invalidate the Data*
      _checkResize();
      // The resize changes the bucket associated with a key
      ht = _add(_hashValueToBucket(hf), id, ValueType());
    }
    return ht;
  }

  /*!
   * \brief Adds the value \a value corresponding to the key \a id
   *
   * If a value corresponding to \a id already exists, the result is
   * undefined.
   */
  void nocheckAdd(KeyTypeConstRef id, const ValueType& value)
  {
    _checkResize();
    Integer hf = _keyToBucket(id);
    _add(hf, id, value);
  }

  ArrayView<Data*> buckets()
  {
    return m_buckets;
  }

  ConstArrayView<Data*> buckets() const
  {
    return m_buckets;
  }

  //! Resizes the hash table
  void resize(Integer new_size, bool use_prime = false)
  {
    if (use_prime)
      new_size = this->nearestPrimeNumber(new_size);
    if (new_size == 0) {
      m_nb_bucket = new_size;
      clear();
      return;
    }
    if (new_size == m_nb_bucket)
      return;
    _rehash(new_size);
  }

  //! Rehashes the data after changing key values
  void rehash()
  {
    _rehash(m_nb_bucket);
  }

 public:

  //! Applies the functor \a f to all elements of the collection
  template <class Lambda> void
  each(const Lambda& lambda)
  {
    for (Integer k = 0, n = m_buckets.size(); k < n; ++k) {
      Data* nbid = m_buckets[k];
      for (; nbid; nbid = nbid->next()) {
        lambda(nbid);
      }
    }
  }

  /*!
   * \brief Applies the functor \a f to all elements of the collection
   * and uses x->value() (of type ValueType) as an argument.
   */
  template <class Lambda> void
  eachValue(const Lambda& lambda)
  {
    for (Integer k = 0, n = m_buckets.size(); k < n; ++k) {
      Data* nbid = m_buckets[k];
      for (; nbid; nbid = nbid->next()) {
        lambda(nbid->value());
      }
    }
  }

 private:

  //! Rehashes the data after changing key values
  void _rehash(Integer new_size)
  {
    //todo: delete the allocation of this array
    UniqueArray<Data*> old_buckets(m_buckets);
    m_count = 0;
    m_nb_bucket = new_size;
    m_buckets.resize(new_size);
    m_buckets.fill(0);
    MultiBufferT<Data>* old_buffer = m_buffer;
    m_first_free = 0;
    m_buffer = new MultiBufferT<Data>(m_nb_bucket);
    for (Integer z = 0, zs = old_buckets.size(); z < zs; ++z) {
      for (Data* i = old_buckets[z]; i; i = i->next()) {
        Data* current = i;
        {
          _add(_keyToBucket(current->key()), current->key(), current->value());
          //Data* new_data = m_buffer->allocOne();
          //new_data->setValue(current->value());
          //_baseAdd(_hash(current->key()),current->key(),new_data);
        }
      }
    }
    delete old_buffer;
    _computeMaxCount();
  }

 private:

  MultiBufferT<Data>* m_buffer; //!< Value allocation buffer
  Data* m_first_free = nullptr; //!< Pointer to the first usable Data

 public:

  mutable Int64 m_nb_collision = 0;
  mutable Int64 m_nb_direct = 0;

 private:

  Data* _add(Integer bucket, KeyTypeConstRef key, const ValueType& value)
  {
    Data* hd = 0;
    if (m_first_free) {
      hd = m_first_free;
      m_first_free = m_first_free->next();
    }
    else
      hd = m_buffer->allocOne();
    hd->setValue(value);
    _baseAdd(bucket, key, hd);
    return hd;
  }

  HashValueType _applyHash(KeyTypeConstRef id) const
  {
    //return (Integer)(KeyTraitsType::hashFunction(id) % m_nb_bucket);
    return KeyTraitsType::hashFunction(id);
  }

  Integer _keyToBucket(KeyTypeConstRef id) const
  {
    return (Integer)(_applyHash(id) % m_nb_bucket);
  }

  Integer _hashValueToBucket(KeyTypeValue id) const
  {
    return (Integer)(id % m_nb_bucket);
  }

  Data* _baseLookupBucket(Integer bucket, KeyTypeConstRef id) const
  {
    for (Data* i = m_buckets[bucket]; i; i = i->next()) {
      if (!(i->key() == id)) {
        ++m_nb_collision;
        continue;
      }
      ++m_nb_direct;
      return i;
    }
    return 0;
  }

  Data* _baseRemoveBucket(Integer bucket, KeyTypeConstRef id)
  {
    Data* i = m_buckets[bucket];
    if (i) {
      if (i->m_key == id) {
        m_buckets[bucket] = i->next();
        --m_count;
        return i;
      }
      for (; i->next(); i = i->next()) {
        if (i->next()->key() == id) {
          Data* r = i->next();
          i->setNext(i->next()->next());
          --m_count;
          return r;
        }
      }
    }
    this->_throwNotFound(id, Printable());
    return 0;
  }

  inline Data* _baseLookup(KeyTypeConstRef id) const
  {
    return _baseLookupBucket(_keyToBucket(id), id);
  }

  inline Data* _baseRemove(KeyTypeConstRef id)
  {
    return _baseRemoveBucket(_keyToBucket(id), id);
  }

  void _baseAdd(Integer bucket, KeyTypeConstRef id, Data* hd)
  {
    Data* buck = m_buckets[bucket];
    hd->m_key = id;
    hd->m_next = buck;
    m_buckets[bucket] = hd;
    ++m_count;
  }

  Data* _lookup(KeyTypeConstRef id)
  {
    return _baseLookup(id);
  }

  const Data* _lookup(KeyTypeConstRef id) const
  {
    return _baseLookup(id);
  }

  Data* _lookupBucket(Integer bucket, KeyTypeConstRef id) const
  {
    return _baseLookupBucket(bucket, id);
  }

  Data* _removeBucket(Integer bucket, KeyTypeConstRef id)
  {
    return _baseRemoveBucket(bucket, id);
  }

  void _checkResize()
  {
    // Resize if necessary.
    if (m_count > m_max_count) {
      //cout << "** BEFORE BUCKET RESIZE this=" << this << " count=" << m_count
      //     << " bucket=" << m_nb_bucket << " m_max_count=" << m_max_count
      //     << " memory=" << (m_buckets.capacity()*sizeof(Data*)) << '\n';
      //_print(Printable());
      // For large tables, increase less quickly to limit
      // memory consumption
      if (m_nb_bucket > 200000) {
        resize((Integer)(1.3 * m_nb_bucket), true);
      }
      else if (m_nb_bucket > 10000) {
        resize((Integer)(1.5 * m_nb_bucket), true);
      }
      else
        resize(2 * m_nb_bucket, true);
      //cout << "** AFTER BUCKET RESIZE this=" << this << " count=" << m_count
      //     << " bucket=" << m_nb_bucket  << " m_max_count=" << m_max_count
      //     << " memory=" << (m_buckets.capacity()*sizeof(Data*)) << '\n';
      //_print(Printable());
      std::cout.flush();
    }
  }

  void _print(FalseType)
  {
  }

  void _print(TrueType)
  {
    for (Integer z = 0, zs = m_buckets.size(); z < zs; ++z) {
      for (Data* i = m_buckets[z]; i; i = i->next()) {
        cout << "* KEY=" << i->key() << " bucket=" << z << '\n';
      }
    }
  }

  void _throwNotFound ARCANE_NORETURN(KeyTypeConstRef, FalseType) const
  {
    HashTableBase::_throwNotFound();
  }

  void _throwNotFound ARCANE_NORETURN(KeyTypeConstRef id, TrueType) const
  {
    std::cout << "ERROR: can not find key=" << id << " bucket=" << _keyToBucket(id) << "\n";
    std::cout.flush();
    HashTableBase::_throwNotFound();
  }

  void _computeMaxCount()
  {
    m_max_count = (Integer)(m_nb_bucket * 0.85);
  }

 private:

  //! Maximum number of elements before resizing
  Integer m_max_count = 0;
  UniqueArray<Data*> m_buckets; //! Array of buckets
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Enumerator for a HashTableMap.
 */
template <typename KeyType, typename ValueType>
class HashTableMapEnumeratorT
{
  typedef HashTableMapT<KeyType, ValueType> HashType;
  typedef typename HashType::Data Data;

 public:

  HashTableMapEnumeratorT(const HashType& rhs)
  : m_buckets(rhs.buckets())
  , m_current_data(0)
  , m_current_bucket(-1)
  {}

 public:

  bool operator++()
  {
    if (m_current_data)
      m_current_data = m_current_data->next();
    if (!m_current_data) {
      while (m_current_data == 0 && (m_current_bucket + 1) < m_buckets.size()) {
        ++m_current_bucket;
        m_current_data = m_buckets[m_current_bucket];
      }
    }
    return m_current_data != 0;
  }
  ValueType& operator*() { return m_current_data->value(); }
  const ValueType& operator*() const { return m_current_data->value(); }
  Data* data() { return m_current_data; }
  const Data* data() const { return m_current_data; }

 public:

  ConstArrayView<Data*> m_buckets;
  Data* m_current_data;
  Integer m_current_bucket;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
