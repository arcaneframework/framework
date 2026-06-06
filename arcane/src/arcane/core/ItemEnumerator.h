// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumerator.h                                            (C) 2000-2025 */
/*                                                                           */
/* Enumerator over mesh entity groups.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMENUMERATOR_H
#define ARCANE_CORE_ITEMENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternalEnumerator.h"
#include "arcane/core/Item.h"
#include "arcane/core/EnumeratorTraceWrapper.h"
#include "arcane/core/IItemEnumeratorTracer.h"
#include "arcane/core/ItemEnumeratorBase.h"
#include "arcane/core/ItemConnectedEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ItemEnumerator.h
 *
 * \brief Types and macros for iterating over mesh entities.
 *
 * This file contains the different enumerator types and macros
 * for iterating over mesh entities.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemEnumeratorCS;
class ItemGroupImpl;
class ItemEnumeratorPOD;

// This method is reserved for SWIG
extern "C++" ARCANE_CORE_EXPORT void _arcaneInternalItemEnumeratorSwigSet(const ItemEnumerator* ie, ItemEnumeratorPOD* vpod);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a list of entities.
 */
class ItemEnumerator
: public ItemEnumeratorBaseT<Item>
{
  friend class ItemEnumeratorCS;
  friend class ItemGroup;
  friend class ItemVector;
  friend class ItemVectorView;
  friend class ItemPairEnumerator;
  template <int Extent> friend class ItemConnectedListView;
  // NOTE: Normally, it would suffice to do this:
  //   template<class T> friend class ItemEnumeratorBase;
  // but this does not work with GCC 8. So we do the specialization
  // manually
  //template<class T> friend class ItemEnumeratorBaseT;
  friend class ItemEnumeratorBaseT<Item>;
  friend class ItemEnumeratorBaseT<Node>;
  friend class ItemEnumeratorBaseT<ItemWithNodes>;
  friend class ItemEnumeratorBaseT<Edge>;
  friend class ItemEnumeratorBaseT<Face>;
  friend class ItemEnumeratorBaseT<Cell>;
  friend class ItemEnumeratorBaseT<Particle>;
  friend class ItemEnumeratorBaseT<DoF>;
  friend ARCANE_CORE_EXPORT void _arcaneInternalItemEnumeratorSwigSet(const ItemEnumerator* ie, ItemEnumeratorPOD* vpod);

 public:

  typedef ItemInternal* ItemInternalPtr;
  using BaseClass = ItemEnumeratorBaseT<Item>;

 public:

  ItemEnumerator() = default;

  ItemEnumerator(const ItemInternalVectorView& view)
  : BaseClass(view, nullptr)
  {}

  ItemEnumerator(const ItemInternalEnumerator& rhs)
  : BaseClass(rhs, true)
  {}

  ItemEnumerator(const impl::ItemIndexedListView<DynExtent>& rhs)
  : BaseClass(rhs)
  {}

 public:

  // For testing
  template <int E> ItemEnumerator(const ItemConnectedListView<E>& rhs)
  : BaseClass(ItemConnectedListViewT<Item, E>(rhs))
  {}

 private:

  // Constructor reserved for ItemGroup
  ItemEnumerator(const ItemInfoListView& items, const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, agroup)
  {}

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumerator(const ItemInternalPtr* items, const Int32* local_ids, Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, n, agroup)
  {}

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumerator(const ItemInternalArrayView& items, const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, agroup)
  {}

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumerator(const ItemInternalVectorView& view, const ItemGroupImpl* agroup)
  : BaseClass(view, agroup)
  {}

 protected:

  // TODO To be removed
  ItemEnumerator(ItemSharedInfo* s, const Int32ConstArrayView& local_ids)
  : BaseClass(s, local_ids)
  {}

  ItemEnumerator(ItemSharedInfo* s, const impl::ItemLocalIdListContainerView& view)
  : BaseClass(s, view)
  {}

 public:

  static ItemEnumerator fromItemEnumerator(const ItemEnumerator& rhs)
  {
    return ItemEnumerator(rhs);
  }

 private:

  ItemEnumerator(const impl::ItemLocalIdListContainerView& view, Int32 index,
                 const ItemGroupImpl* agroup, Item item_base)
  : BaseClass(view, index, agroup, item_base)
  {}
  ItemSharedInfo* _sharedInfo() const { return _internalItemBase().m_shared_info; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructor only used by fromItemEnumerator()
inline ItemEnumeratorBase::
ItemEnumeratorBase(const ItemEnumerator& rhs, bool)
: m_view(rhs.m_view)
, m_index(rhs.index())
, m_group_impl(rhs.group())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructor only used by fromItemEnumerator()
template <typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemEnumerator& rhs, bool v)
: ItemEnumeratorBase(rhs, v)
, m_item(rhs._internalItemBase())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemEnumeratorBase::
ItemEnumeratorBase(const ItemEnumerator& rhs)
: m_view(rhs.m_view)
, m_index(rhs.index())
, m_group_impl(rhs.group())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemEnumerator& rhs)
: ItemEnumeratorBase(rhs)
, m_item(rhs._internalItemBase())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemInternalEnumerator& rhs)
: ItemEnumeratorBaseT(ItemEnumerator(rhs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemEnumerator ItemEnumeratorBaseT<ItemType>::
toItemEnumerator() const
{
  return ItemEnumerator(m_view, m_index, m_group_impl, m_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a typed list of entities of type \a ItemType
 */
template <typename ItemType>
class ItemEnumeratorT
: public ItemEnumeratorBaseT<ItemType>
{
 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;
  using BaseClass = ItemEnumeratorBaseT<ItemType>;
  friend class ItemVectorT<ItemType>;
  friend class ItemVectorViewT<ItemType>;
  friend class ItemConnectedListViewT<ItemType>;
  friend class SimdItemEnumeratorT<ItemType>;
  template <typename I1, typename I2> friend class ItemPairEnumeratorT;

 public:

  ItemEnumeratorT() = default;
  ItemEnumeratorT(const ItemVectorView& rhs)
  : BaseClass(rhs)
  {}
  ItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs)
  : BaseClass(rhs)
  {}

 public:

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemEnumerator& rhs)
  : BaseClass(rhs)
  {}

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemInternalEnumerator& rhs)
  : BaseClass(rhs)
  {}

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumeratorT(const ItemInternalPtr* items, const Int32* local_ids, Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, n, agroup)
  {}

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumeratorT(const ItemInternalArrayView& items, const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, agroup)
  {}

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumeratorT(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view, agroup)
  {}

 public:

  // For testing
  ItemEnumeratorT(const ItemConnectedListViewT<ItemType>& rhs)
  : BaseClass(rhs)
  {}

 private:

  // Constructor reserved for ItemGroup
  ItemEnumeratorT(const ItemInfoListViewT<ItemType>& items, const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, agroup)
  {}

 private:

  // TODO: to be removed
  ItemEnumeratorT(ItemSharedInfo* s, const Int32ConstArrayView& local_ids)
  : BaseClass(s, local_ids)
  {}

  ItemEnumeratorT(ItemSharedInfo* s, const impl::ItemLocalIdListContainerView& view)
  : BaseClass(s, view)
  {}

 public:

  //! Conversion to an ItemEnumerator
  operator ItemEnumerator() const { return this->toItemEnumerator(); }

 public:

  static ItemEnumeratorT<ItemType> fromItemEnumerator(const ItemEnumerator& rhs)
  {
    return ItemEnumeratorT<ItemType>(rhs, true);
  }

 private:

  //! Constructor only used by fromItemEnumerator()
  ItemEnumeratorT(const ItemEnumerator& rhs, bool v)
  : BaseClass(rhs, v)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemEnumerator ItemVectorView::
enumerator() const
{
  return ItemEnumerator(m_shared_info, m_index_view._localIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int Extent>
inline ItemEnumerator ItemConnectedListView<Extent>::
enumerator() const
{
  return ItemEnumerator(m_shared_info, m_index_view._localIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemLocalId::
ItemLocalId(ItemEnumeratorT<ItemType> enumerator)
: m_local_id(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId::
ItemLocalId(ItemConnectedEnumerator enumerator)
: m_local_id(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemLocalId::
ItemLocalId(ItemConnectedEnumeratorT<ItemType> enumerator)
: m_local_id(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: add type checking
template <typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemEnumerator enumerator)
: ItemLocalId(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemConnectedEnumeratorT<ItemType> enumerator)
: ItemLocalId(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_CHECK_ENUMERATOR(enumerator, testgroup) \
  ARCANE_ASSERT(((enumerator).group() == (testgroup).internal()), ("Invalid access on partial data using enumerator not associated to underlying group %s", testgroup.name().localstr()))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define A_ENUMERATE_ITEM(_EnumeratorClassName, iname, view) \
  for (A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) iname(_EnumeratorClassName ::fromItemEnumerator((view).enumerator()) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname)

#define A_ENUMERATE_ITEM_NO_TRACE(_EnumeratorClassName, iname, view) \
  for (_EnumeratorClassName iname(_EnumeratorClassName ::fromItemEnumerator((view).enumerator())); iname.hasNext(); ++iname)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Generic enumerator for an entity group
#define ENUMERATE_NO_TRACE_(type, name, group) A_ENUMERATE_ITEM_NO_TRACE(::Arcane::ItemEnumeratorT<type>, name, group)

//! Generic enumerator for an entity group
#define ENUMERATE_(type, name, group) A_ENUMERATE_ITEM(::Arcane::ItemEnumeratorT<type>, name, group)

//! Generic enumerator for an entity group
#define ENUMERATE_GENERIC(type, name, group) A_ENUMERATE_ITEM(::Arcane::ItemEnumeratorT<type>, name, group)

//! Generic enumerator for a node group
#define ENUMERATE_ITEM(name, group) A_ENUMERATE_ITEM(::Arcane::ItemEnumerator, name, group)

#define ENUMERATE_ITEMWITHNODES(name, group) ENUMERATE_ (::Arcane::ItemWithNodes, name, group)

//! Generic enumerator for a node group
#define ENUMERATE_NODE(name, group) ENUMERATE_ (::Arcane::Node, name, group)

//! Generic enumerator for an edge group
#define ENUMERATE_EDGE(name, group) ENUMERATE_ (::Arcane::Edge, name, group)

//! Generic enumerator for a face group
#define ENUMERATE_FACE(name, group) ENUMERATE_ (::Arcane::Face, name, group)

//! Generic enumerator for a cell group
#define ENUMERATE_CELL(name, group) ENUMERATE_ (::Arcane::Cell, name, group)

//! Generic enumerator for a particle group
#define ENUMERATE_PARTICLE(name, group) ENUMERATE_ (::Arcane::Particle, name, group)

//! Generic enumerator for a degree of freedom group
#define ENUMERATE_DOF(name, group) ENUMERATE_ (::Arcane::DoF, name, group)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over an ItemPairGroup.
 * \param _item_type1 Type of the group entity
 * \param _item_type2 Type of the sub-entities of the group
 * \param _name Name of the enumerator
 * \param _array Instance of ItemPairGroup
 */
#define ENUMERATE_ITEMPAIR(_item_type1, _item_type2, _name, _array) \
  for (::Arcane::ItemPairEnumeratorT<_item_type1, _item_type2> _name(_array); _name.hasNext(); ++_name)

/*!
 * \brief Generic enumerator over an ItemPairGroup.
 * \sa ENUMERATE_ITEMPAIR
 */
#define ENUMERATE_ITEMPAIR_DIRECT(_name, _array) \
  for (::Arcane::ItemPairEnumerator _name(_array); _name.hasNext(); ++_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a sub-element of an ItemPairGroup.
 * \param _item_type Type of the sub-entity
 * \param _name Name of the enumerator
 * \param _parent_item Instance of the parent entity or the enumerator
 * on the parent entity.
 */
#define ENUMERATE_SUB_ITEM(_item_type, _name, _parent_item) \
  for (::Arcane::ItemEnumeratorT<_item_type> _name(_parent_item.subItems()); _name.hasNext(); ++_name)

/*!
 * \brief Generic enumerator over a sub-element of an ItemPairGroup.
 * \sa ENUMERATE_SUB_ITEM
 */
#define ENUMERATE_SUB_ITEM_DIRECT(_name, _parent_item) \
  for (::Arcane::ItemInternalEnumerator _name(_parent_item.subItems()); _name.hasNext(); ++_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
