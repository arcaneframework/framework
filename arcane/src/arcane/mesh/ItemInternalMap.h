// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalMap.h                                           (C) 2000-2025 */
/*                                                                           */
/* Associative array of ItemInternal.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMINTERNALMAP_H
#define ARCANE_MESH_ITEMINTERNALMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/utils/HashTableMap2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Associative array of ItemInternal.
 *
 * This class is internal to Arcane.
 *
 * The key of this associative array is the UniqueId of the entities.
 * If it changes, you must call notifyUniqueIdsChanged() to update
 * the associative array.
 *
 * \note All methods that use or return an 'ItemInternal*'
 * are obsolete and should not be used.
 */
class ARCANE_MESH_EXPORT ItemInternalMap
{
  // For access to methods that use ItemInternal.
  friend class DynamicMeshKindInfos;

 private:

  using LegacyImpl = HashTableMapT<Int64, ItemInternal*>;
  using NewImpl = impl::HashTableMap2<Int64, ItemInternal*>;
  using BaseData = LegacyImpl::Data;

 public:

  static constexpr bool UseNewImpl = 1;

  class LookupData
  {
    friend ItemInternalMap;

   public:

    void setValue(ItemInternal* v)
    {
      m_value = v;
      if (m_legacy_data)
        m_legacy_data->setValue(v);
      else
        m_iter->second = v;
    }
    ItemInternal* value() const { return m_value; }

   private:

    explicit LookupData(NewImpl::iterator x)
    : m_iter(x)
    , m_value(x->second)
    {}
    explicit LookupData(BaseData* d)
    : m_legacy_data(d)
    , m_value(d->value())
    {}
    NewImpl::iterator m_iter;
    BaseData* m_legacy_data = nullptr;
    ItemInternal* m_value;
  };

 public:

  using Data ARCANE_DEPRECATED_REASON("Y2024: Data type is internal to Arcane") = LegacyImpl::Data;

 public:

  using ValueType = ItemInternal*;

 public:

  ItemInternalMap();

 public:

  /*!
   * \brief Adds the value \a v corresponding to the key \a key
   *
   * If a value corresponding to \a id already exists, it is replaced.
   *
   * \retval true if the key is added
   * \retval false if the key already exists and is replaced
   */
  bool add(Int64 key, ItemInternal* v)
  {
    return m_new_impl.insert(std::make_pair(key, v)).second;
  }

  //! Removes all elements from the table
  void clear()
  {
    return m_new_impl.clear();
  }

  //! Number of elements in the table
  Int32 count() const
  {
    return CheckedConvert::toInt32(m_new_impl.size());
  }

  /*!
   * \brief Removes the value associated with the key \a key
   *
   * Throws an exception if there are no values associated with the key
   */
  void remove(Int64 key)
  {
    auto x = m_new_impl.find(key);
    if (x == m_new_impl.end())
      _throwNotFound(key);
    m_new_impl.erase(x);
  }

  //! \a true if a value with the key \a id is present
  bool hasKey(Int64 key)
  {
    return (m_new_impl.find(key) != m_new_impl.end());
  }

  //! Resizes the hash table
  void resize([[maybe_unused]] Int32 new_size, [[maybe_unused]] bool use_prime = false)
  {
  }

  /*!
   * \brief Notifies that the unique IDs of the entities have changed.
   *
   * This call may cause a complete recalculation of the associative array.
   */
  void notifyUniqueIdsChanged();

  /*!
   * \brief Template function to iterate over the instance's entities.
   *
   * The type of the template argument can be any type of entity
   * that can be constructed from an impl::ItemBase.
   * \code
   * ItemInternalMap item_map = ...
   * item_map.eachItem([&](Item item){
   *   std::cout << "LID=" << item_base.localId() << "\n";
   * });
   * \endcode
   */
  template <class Lambda> void
  eachItem(const Lambda& lambda)
  {
    for (auto [key, value] : m_new_impl)
      lambda(Arcane::impl::ItemBase(value));
  }
  //! Number of buckets
  Int32 nbBucket() const
  {
    return CheckedConvert::toInt32(m_new_impl.bucket_count());
  }

 public:

  //! Returns the entity associated with \a key if found, or the null entity otherwise
  impl::ItemBase tryFind(Int64 key) const
  {
    auto x = m_new_impl.find(key);
    return (x != m_new_impl.end()) ? x->second : impl::ItemBase{};
  }
  //! Returns the localId() associated with \a key if found, or none otherwise
  Int32 tryFindLocalId(Int64 key) const
  {
    auto x = m_new_impl.find(key);
    return (x != m_new_impl.end()) ? x->second->localId() : NULL_ITEM_LOCAL_ID;
  }

  /*!
   * \brief Returns the unique ID entity \a uid.
   *
   * Throws an exception if the entity is not in the table.
   */
  impl::ItemBase findItem(Int64 uid) const
  {
    auto x = m_new_impl.find(uid);
    if (x == m_new_impl.end())
      _throwNotFound(uid);
    return x->second;
  }

  /*!
   * \brief Returns the local number of the unique ID entity \a uid.
   *
   * Throws an exception if the entity is not in the table.
   */
  Int32 findLocalId(Int64 uid) const
  {
    auto x = m_new_impl.find(uid);
    if (x == m_new_impl.end())
      _throwNotFound(uid);
    return x->second->localId();
  }

  void checkValid() const;

 public:

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  Data* lookup([[maybe_unused]] Int64 key)
  {
    _throwNotSupported("lookup");
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  const Data* lookup([[maybe_unused]] Int64 key) const
  {
    _throwNotSupported("lookup");
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  ConstArrayView<BaseData*> buckets() const
  {
    _throwNotSupported("buckets");
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  BaseData* lookupAdd([[maybe_unused]] Int64 id,
                      [[maybe_unused]] ItemInternal* value,
                      [[maybe_unused]] bool& is_add)
  {
    _throwNotSupported("lookupAdd(id,value,is_add)");
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  BaseData* lookupAdd([[maybe_unused]] Int64 uid)
  {
    _throwNotSupported("lookupAdd(uid)");
  }

  ARCANE_DEPRECATED_REASON("Y2024: Use findItem() instead")
  ItemInternal* lookupValue([[maybe_unused]] Int64 uid) const
  {
    _throwNotSupported("lookupValue");
  }

  ARCANE_DEPRECATED_REASON("Y2024: Use findItem() instead")
  ItemInternal* operator[]([[maybe_unused]] Int64 uid) const
  {
    _throwNotSupported("operator[]");
  }

 private:

  NewImpl m_new_impl;

 private:

  // The following three methods are only for the
  // class DynamicMeshKindInfos.

  //! Changes the values of localId()
  void _changeLocalIds(ArrayView<ItemInternal*> items_internal,
                       ConstArrayView<Int32> old_to_new_local_ids);

  LookupData _lookupAdd(Int64 id, ItemInternal* value, bool& is_add)
  {
    auto x = m_new_impl.insert(std::make_pair(id, value));
    is_add = x.second;
    return LookupData(x.first);
  }

  //! Returns the entity associated with \a key if found, or nullptr otherwise
  ItemInternal* _tryFindItemInternal(Int64 key) const
  {
    auto x = m_new_impl.find(key);
    if (x == m_new_impl.end())
      return nullptr;
    _checkValid(key, x->second);
    return x->second;
  }

 private:

  void _throwNotFound ARCANE_NORETURN(Int64 id) const;
  void _throwNotSupported ARCANE_NORETURN(const char* func_name) const;
  void _checkValid(Int64 uid, ItemInternal* v) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
