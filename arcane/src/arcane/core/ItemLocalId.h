// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalId.h                                               (C) 2000-2023 */
/*                                                                           */
/* Index local sur une entité du maillage.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMLOCALID_H
#define ARCANE_ITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
class MeshUnitTest;
}

namespace Arcane
{
namespace mesh
{
  class IndexedItemConnectivityAccessor;
}

// TODO: rendre obsolète les constructeurs qui prennent un argument
// un ItemEnumerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Item dans une variable.
 */
class ARCANE_CORE_EXPORT ItemLocalId
{
 public:

  ItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalId(Int32 id)
  : m_local_id(id)
  {}
  // La définition de ce constructeur est dans ItemInternal.h
  inline ItemLocalId(ItemInternal* item);
  inline ItemLocalId(ItemEnumerator enumerator);
  template <typename ItemType> inline ItemLocalId(ItemEnumeratorT<ItemType> enumerator);
  inline ItemLocalId(Item item);
  constexpr ARCCORE_HOST_DEVICE operator Int32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInt32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInteger() const { return m_local_id; }

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 localId() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_local_id == NULL_ITEM_LOCAL_ID; }

 public:

  static SmallSpan<const ItemLocalId> fromSpanInt32(SmallSpan<const Int32> v)
  {
    auto* ptr = reinterpret_cast<const ItemLocalId*>(v.data());
    return { ptr, v.size() };
  }
  static SmallSpan<const Int32> toSpanInt32(SmallSpan<const ItemLocalId> v)
  {
    auto* ptr = reinterpret_cast<const Int32*>(v.data());
    return { ptr, v.size() };
  }

 private:

  Int32 m_local_id = NULL_ITEM_LOCAL_ID;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une entité \a ItemType dans une variable.
 */
template <typename ItemType>
class ItemLocalIdT
: public ItemLocalId
{
 public:

  using ThatClass = ItemLocalIdT<ItemType>;

 public:

  ItemLocalIdT() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalIdT(Int32 id)
  : ItemLocalId(id)
  {}
  inline ItemLocalIdT(ItemInternal* item);
  inline ItemLocalIdT(ItemEnumeratorT<ItemType> enumerator);
  inline ItemLocalIdT(ItemType item);

 public:

  static SmallSpan<const ItemLocalId> fromSpanInt32(SmallSpan<const Int32> v)
  {
    auto* ptr = reinterpret_cast<const ThatClass*>(v.data());
    return { ptr, v.size() };
  }

  static SmallSpan<const Int32> toSpanInt32(SmallSpan<const ThatClass> v)
  {
    auto* ptr = reinterpret_cast<const Int32*>(v.data());
    return { ptr, v.size() };
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use strongly typed 'ItemEnumeratorT<ItemType>' or 'ItemType'")
  inline ItemLocalIdT(ItemEnumerator enumerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue typée sur une liste d'entités d'une connectivité.
 */
template <typename ItemType>
class ItemLocalIdViewT
{
  friend class ItemConnectivityContainerView;
  friend mesh::IndexedItemConnectivityAccessor;
  friend ArcaneTest::MeshUnitTest;
  friend class Item;

 public:

  using LocalIdType = typename ItemLocalIdTraitsT<ItemType>::LocalIdType;
  using SpanType = SmallSpan<const LocalIdType>;
  using iterator = typename SpanType::iterator;
  using const_iterator = typename SpanType::const_iterator;

 public:

  ItemLocalIdViewT() = default;

 private:

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdViewT(SpanType ids)
  : m_ids(ids)
  {}
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdViewT(const LocalIdType* ids, Int32 s)
  : m_ids(ids, s)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE LocalIdType operator[](Int32 i) const { return m_ids[i]; }
  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_ids.size(); }
  constexpr ARCCORE_HOST_DEVICE const_iterator begin() const { return m_ids.begin(); }
  constexpr ARCCORE_HOST_DEVICE const_iterator end() const { return m_ids.end(); }

 public:
 private:

  static ARCCORE_HOST_DEVICE ItemLocalIdViewT<ItemType>
  fromIds(SmallSpan<const Int32> v)
  {
    return ItemLocalIdViewT<ItemType>(reinterpret_cast<const LocalIdType*>(v.data()), v.size());
  }

  ConstArrayView<Int32> toViewInt32() const
  {
    return { size(), reinterpret_cast<const Int32*>(data()) };
  }
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> toSpanInt32() const
  {
    return { reinterpret_cast<const Int32*>(data()), size() };
  }
  constexpr ARCCORE_HOST_DEVICE const LocalIdType* data() const { return m_ids.data(); }
  constexpr ARCCORE_HOST_DEVICE SpanType ids() const { return m_ids; }

 private:

  SpanType m_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
