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
#ifndef ARCANE_CORE_ITEMLOCALID_H
#define ARCANE_CORE_ITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  inline ItemLocalId(ItemConnectedEnumerator enumerator);
  template <typename ItemType> inline ItemLocalId(ItemEnumeratorT<ItemType> enumerator);
  template <typename ItemType> inline ItemLocalId(ItemConnectedEnumeratorT<ItemType> enumerator);
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
  inline ItemLocalIdT(ItemConnectedEnumeratorT<ItemType> enumerator);
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
