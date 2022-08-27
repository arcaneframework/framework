// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalId.h                                               (C) 2000-2022 */
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

namespace Arcane
{
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
  constexpr ARCCORE_HOST_DEVICE ItemLocalId() : m_local_id(NULL_ITEM_LOCAL_ID){}
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalId(Int32 id) : m_local_id(id){}
  // La définition de ce constructeur est dans ItemInternal.h
  inline ItemLocalId(ItemInternal* item);
  inline ItemLocalId(ItemEnumerator enumerator);
  template<typename ItemType> inline ItemLocalId(ItemEnumeratorT<ItemType> enumerator);
  inline ItemLocalId(Item item);
  constexpr ARCCORE_HOST_DEVICE operator Int32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInt32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInteger() const { return m_local_id; }
 public:
  constexpr ARCCORE_HOST_DEVICE Int32 localId() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_local_id==NULL_ITEM_LOCAL_ID; }
 private:
  Int32 m_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une entité \a ItemType dans une variable.
 */
template<typename ItemType> class ItemLocalIdT
: public ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalIdT(Int32 id) : ItemLocalId(id){}
  inline ItemLocalIdT(ItemInternal* item);
  inline ItemLocalIdT(ItemEnumeratorT<ItemType> enumerator);
  inline ItemLocalIdT(ItemType item);
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
 public:
  using LocalIdType = typename ItemLocalIdTraitsT<ItemType>::LocalIdType;
  using SpanType = SmallSpan<const LocalIdType>;
  using iterator = typename SpanType::iterator;
  using const_iterator = typename SpanType::const_iterator;
 public:
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdViewT(SpanType ids) : m_ids(ids){}
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdViewT(const LocalIdType* ids,Int32 s) : m_ids(ids,s){}
  ItemLocalIdViewT() = default;
  constexpr ARCCORE_HOST_DEVICE operator SpanType() const { return m_ids; }
 public:
  constexpr ARCCORE_HOST_DEVICE SpanType ids() const { return m_ids; }
  constexpr ARCCORE_HOST_DEVICE LocalIdType operator[](Int32 i) const { return m_ids[i]; }
  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_ids.size(); }
  constexpr ARCCORE_HOST_DEVICE iterator begin() { return m_ids.begin(); }
  constexpr ARCCORE_HOST_DEVICE iterator end() { return m_ids.end(); }
  constexpr ARCCORE_HOST_DEVICE const_iterator begin() const { return m_ids.begin(); }
  constexpr ARCCORE_HOST_DEVICE const_iterator end() const { return m_ids.end(); }
 public:
  constexpr ARCCORE_HOST_DEVICE const LocalIdType* data() const { return m_ids.data(); }
 public:
  static constexpr ARCCORE_HOST_DEVICE ItemLocalIdViewT<ItemType>
  fromIds(SmallSpan<const Int32> v)
  {
    return ItemLocalIdViewT<ItemType>(reinterpret_cast<const LocalIdType*>(v.data()),v.size());
  }
 private:
  SpanType m_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
