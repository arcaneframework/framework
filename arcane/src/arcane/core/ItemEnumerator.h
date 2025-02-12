// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumerator.h                                            (C) 2000-2025 */
/*                                                                           */
/* Enumérateur sur des groupes d'entités du maillage.                        */
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
 * \brief Types et macros pour itérer sur les entités du maillage.
 *
 * Ce fichier contient les différentes types d'itérateur et les macros
 * pour itérer sur les entités du maillage.
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

// Cette méthode est réservée pour SWIG
extern "C++" ARCANE_CORE_EXPORT
void _arcaneInternalItemEnumeratorSwigSet(const ItemEnumerator* ie, ItemEnumeratorPOD* vpod);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste d'entités.
 */
class ItemEnumerator
: public ItemEnumeratorBaseT<Item>
{
  friend class ItemEnumeratorCS;
  friend class ItemGroup;
  friend class ItemVector;
  friend class ItemVectorView;
  friend class ItemPairEnumerator;
  template<int Extent> friend class ItemConnectedListView;
  // NOTE: Normalement il suffirait de faire cela:
  //   template<class T> friend class ItemEnumeratorBase;
  // mais cela ne fonctionne pas avec GCC 8. On fait donc la spécialisation
  // à la main
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

  // Pour test
  template<int E> ItemEnumerator(const ItemConnectedListView<E>& rhs)
  : BaseClass(ItemConnectedListViewT<Item,E>(rhs)){}

 private:

  // Constructeur réservé pour ItemGroup
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

  // TODO A supprimer
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

//! Constructeur seulement utilisé par fromItemEnumerator()
inline ItemEnumeratorBase::
ItemEnumeratorBase(const ItemEnumerator& rhs, bool)
: m_view(rhs.m_view)
, m_index(rhs.index())
, m_group_impl(rhs.group())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructeur seulement utilisé par fromItemEnumerator()
template<typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemEnumerator& rhs,bool v)
: ItemEnumeratorBase(rhs,v)
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

template<typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemEnumerator& rhs)
: ItemEnumeratorBase(rhs)
, m_item(rhs._internalItemBase())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemInternalEnumerator& rhs)
: ItemEnumeratorBaseT(ItemEnumerator(rhs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemEnumerator ItemEnumeratorBaseT<ItemType>::
toItemEnumerator() const
{
  return ItemEnumerator(m_view,m_index,m_group_impl,m_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste typée d'entités de type \a ItemType
 */
template<typename ItemType>
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
  ItemEnumeratorT(const ItemVectorView& rhs) : BaseClass(rhs){}
  ItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs) : BaseClass(rhs){}

 public:

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemEnumerator& rhs)
  : BaseClass(rhs){}

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemInternalEnumerator& rhs)
  : BaseClass(rhs){}

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup){}

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumeratorT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}

  ARCANE_DEPRECATED_REASON("Y2022: Internal to Arcane. Use other constructor")
  ItemEnumeratorT(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view,agroup){}

 public:

  // Pour test
  ItemEnumeratorT(const ItemConnectedListViewT<ItemType>& rhs) : BaseClass(rhs){}

 private:

  // Constructeur réservé pour ItemGroup
  ItemEnumeratorT(const ItemInfoListViewT<ItemType>& items, const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items, local_ids, agroup)
  {}

 private:

  // TODO: a supprimer
  ItemEnumeratorT(ItemSharedInfo* s,const Int32ConstArrayView& local_ids)
  : BaseClass(s,local_ids){}

  ItemEnumeratorT(ItemSharedInfo* s,const impl::ItemLocalIdListContainerView& view)
  : BaseClass(s,view){}

 public:

  //! Conversion vers un ItemEnumerator
  operator ItemEnumerator() const { return this->toItemEnumerator(); }

 public:

  static ItemEnumeratorT<ItemType> fromItemEnumerator(const ItemEnumerator& rhs)
  {
    return ItemEnumeratorT<ItemType>(rhs,true);
  }

 private:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorT(const ItemEnumerator& rhs,bool v) : BaseClass(rhs,v){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemEnumerator ItemVectorView::
enumerator() const
{
  return ItemEnumerator(m_shared_info,m_index_view._localIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int Extent>
inline ItemEnumerator ItemConnectedListView<Extent>::
enumerator() const
{
  return ItemEnumerator(m_shared_info,m_index_view._localIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemLocalId::
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

template<typename ItemType> inline ItemLocalId::
ItemLocalId(ItemConnectedEnumeratorT<ItemType> enumerator)
: m_local_id(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: ajouter vérification du bon type
template<typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemEnumerator enumerator)
: ItemLocalId(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemConnectedEnumeratorT<ItemType> enumerator)
: ItemLocalId(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_CHECK_ENUMERATOR(enumerator,testgroup)                   \
  ARCANE_ASSERT(((enumerator).group()==(testgroup).internal()),("Invalid access on partial data using enumerator not associated to underlying group %s",testgroup.name().localstr()))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define A_ENUMERATE_ITEM(_EnumeratorClassName,iname,view)               \
  for( A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) iname(_EnumeratorClassName :: fromItemEnumerator((view).enumerator()) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname )

#define A_ENUMERATE_ITEM_NO_TRACE(_EnumeratorClassName,iname,view)               \
  for( _EnumeratorClassName iname(_EnumeratorClassName :: fromItemEnumerator((view).enumerator())); iname.hasNext(); ++iname )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumérateur générique d'un groupe d'entité
#define ENUMERATE_NO_TRACE_(type,name,group) A_ENUMERATE_ITEM_NO_TRACE(::Arcane::ItemEnumeratorT< type >,name,group)

//! Enumérateur générique d'un groupe d'entité
#define ENUMERATE_(type,name,group) A_ENUMERATE_ITEM(::Arcane::ItemEnumeratorT< type >,name,group)

//! Enumérateur générique d'un groupe d'entité
#define ENUMERATE_GENERIC(type,name,group) A_ENUMERATE_ITEM(::Arcane::ItemEnumeratorT< type >,name,group)

//! Enumérateur générique d'un groupe de noeuds
#define ENUMERATE_ITEM(name,group) A_ENUMERATE_ITEM(::Arcane::ItemEnumerator,name,group)

#define ENUMERATE_ITEMWITHNODES(name,group) ENUMERATE_(::Arcane::ItemWithNodes,name,group)

//! Enumérateur générique d'un groupe de noeuds
#define ENUMERATE_NODE(name,group) ENUMERATE_(::Arcane::Node,name,group)

//! Enumérateur générique d'un groupe d'arêtes
#define ENUMERATE_EDGE(name,group) ENUMERATE_(::Arcane::Edge,name,group)

//! Enumérateur générique d'un groupe de faces
#define ENUMERATE_FACE(name,group) ENUMERATE_(::Arcane::Face,name,group)

//! Enumérateur générique d'un groupe de mailles
#define ENUMERATE_CELL(name,group) ENUMERATE_(::Arcane::Cell,name,group)

//! Enumérateur générique d'un groupe de particules
#define ENUMERATE_PARTICLE(name,group) ENUMERATE_(::Arcane::Particle,name,group)

//! Enumérateur generique d'un groupe de degrés de liberté
#define ENUMERATE_DOF(name,group) ENUMERATE_(::Arcane::DoF,name,group)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumérateur sur un ItemPairGroup.
 * \param _item_type1 Type de l'entité du groupe
 * \param _item_type2 Type des sous-entités du groupe
 * \param _name Nom de l'énumérateur
 * \param _group Instance de ItemPairGroup
 */
#define ENUMERATE_ITEMPAIR(_item_type1,_item_type2,_name,_array) \
for( ::Arcane::ItemPairEnumeratorT< _item_type1, _item_type2 > _name(_array); _name.hasNext(); ++_name )

/*!
 * \brief Enumérateur générique sur un ItemPairGroup.
 * \sa ENUMERATE_ITEMPAIR
 */
#define ENUMERATE_ITEMPAIR_DIRECT(_name,_array) \
for( ::Arcane::ItemPairEnumerator _name(_array); _name.hasNext(); ++_name )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumérateur sur sous-élément d'un ItemPairGroup.
 * \param _item_type Type de la sous-entité
 * \param _name Nom de l'énumérateur
 * \param _parent_item Instance de l'entité parente ou de l'énumérateur
 * sur l'entité parente.
 */
#define ENUMERATE_SUB_ITEM(_item_type,_name,_parent_item) \
for( ::Arcane::ItemEnumeratorT< _item_type > _name(_parent_item.subItems()); _name.hasNext(); ++_name )

/*!
 * \brief Enumérateur générique sur un sous-élément d'un ItemPairGroup.
 * \sa ENUMERATE_SUB_ITEM
 */
#define ENUMERATE_SUB_ITEM_DIRECT(_name,_parent_item) \
for( ::Arcane::ItemInternalEnumerator _name(_parent_item.subItems()); _name.hasNext(); ++_name )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
