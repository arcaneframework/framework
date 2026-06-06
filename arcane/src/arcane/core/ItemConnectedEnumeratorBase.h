// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectedEnumeratorBase.h                               (C) 2000-2025 */
/*                                                                           */
/* Base class for enumerators over connected entities of the mesh.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMCONNECTEDENUMERATORBASE_H
#define ARCANE_CORE_ITEMCONNECTEDENUMERATORBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternalEnumerator.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for enumerators over a list of connected entities.
 *
 * Instances of this class are created either via ItemConnectedEnumerator,
 * or via ItemConnectedEnumeratorT.
 *
 * \code
 * for( ItemConnectedEnumeratorBase iter(...); iter.hasNext(); ++iter )
 *   ;
 * \endcode
 */
class ItemConnectedEnumeratorBase
{
  // Only these classes are allowed to construct instances of this class
  template <typename T> friend class ItemConnectedEnumeratorBaseT;

 private:

  ItemConnectedEnumeratorBase() = default;
  explicit ItemConnectedEnumeratorBase(const ConstArrayView<Int32> local_ids)
  : m_local_ids(local_ids.data())
  , m_count(local_ids.size())
  {}
  template <int E> explicit ItemConnectedEnumeratorBase(const ItemConnectedListView<E>& rhs)
  : m_local_ids(rhs._localIds().data())
  , m_count(rhs._localIds().size())
  , m_local_id_offset(rhs._localIdOffset())
  {}
  ItemConnectedEnumeratorBase(const Int32* local_ids, Int32 index, Int32 n)
  : m_local_ids(local_ids)
  , m_index(index)
  , m_count(n)
  {
  }

 public:

  //! Increments the enumerator index
  constexpr void operator++()
  {
    ++m_index;
  }

  //! True if the end of the enumerator has not been reached (index()<count())
  constexpr bool operator()() const
  {
    return m_index < m_count;
  }

  //! True if the end of the enumerator has not been reached (index()<count())
  constexpr bool hasNext() const { return m_index < m_count; }

  //! Number of elements in the enumerator
  constexpr Int32 count() const { return m_count; }

  //! Current index of the enumerator
  constexpr Int32 index() const { return m_index; }

  //! localId() of the current entity.
  constexpr ItemLocalId itemLocalId() const { return ItemLocalId(m_local_id_offset + m_local_ids[m_index]); }

  //! localId() of the current entity.
  constexpr Int32 localId() const { return m_local_id_offset + m_local_ids[m_index]; }

 protected:

  const Int32* ARCANE_RESTRICT m_local_ids = nullptr;
  Int32 m_index = 0;
  Int32 m_count = 0;
  Int32 m_local_id_offset = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Typed base class for enumerators over a list of connected entities.
 *
 * Instances of this class are created either via ItemConnectedEnumerator, or
 * via ItemConnectedEnumeratorT.
 */
template <typename ItemType>
class ItemConnectedEnumeratorBaseT
: public ItemConnectedEnumeratorBase
{
  friend class ItemConnectedEnumerator;
  friend class ItemConnectedEnumeratorT<ItemType>;

 private:

  using LocalIdType = typename ItemType::LocalIdType;
  using BaseClass = ItemConnectedEnumeratorBase;

 private:

  ItemConnectedEnumeratorBaseT()
  : BaseClass()
  , m_item(NULL_ITEM_LOCAL_ID, ItemSharedInfo::nullInstance())
  {}

  ItemConnectedEnumeratorBaseT(ItemSharedInfo* shared_info, const Int32ConstArrayView& local_ids)
  : BaseClass(local_ids)
  , m_item(NULL_ITEM_LOCAL_ID, shared_info)
  {}

  ItemConnectedEnumeratorBaseT(const impl::ItemIndexedListView<DynExtent>& view)
  : ItemConnectedEnumeratorBaseT(view.m_shared_info, view.constLocalIds())
  {}

  ItemConnectedEnumeratorBaseT(const ItemConnectedListViewT<ItemType>& rhs)
  : BaseClass(rhs)
  , m_item(NULL_ITEM_LOCAL_ID, rhs.m_shared_info)
  {}

  ItemConnectedEnumeratorBaseT(const Int32* local_ids, Int32 index, Int32 n, Item item_base)
  : ItemConnectedEnumeratorBase(local_ids, index, n)
  , m_item(item_base)
  {
  }

 public:

  constexpr ItemType operator*() const
  {
    m_item.m_local_id = m_local_id_offset + m_local_ids[m_index];
    return m_item;
  }
  constexpr const ItemType* operator->() const
  {
    m_item.m_local_id = m_local_id_offset + m_local_ids[m_index];
    return &m_item;
  }

  constexpr LocalIdType asItemLocalId() const
  {
    return LocalIdType{ m_local_id_offset + m_local_ids[m_index] };
  }

 protected:

  mutable ItemType m_item = ItemType(NULL_ITEM_LOCAL_ID, nullptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
