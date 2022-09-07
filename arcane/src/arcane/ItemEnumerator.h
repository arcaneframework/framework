// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumerator.h                                            (C) 2000-2022 */
/*                                                                           */
/* Enumérateur sur des groupes d'entités du maillage.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMENUMERATOR_H
#define ARCANE_ITEMENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternalEnumerator.h"
#include "arcane/Item.h"
#include "arcane/EnumeratorTraceWrapper.h"
#include "arcane/IItemEnumeratorTracer.h"
#include "arcane/ItemEnumeratorBase.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste d'entités.
 */
class ItemEnumerator
: public ItemEnumeratorBaseT<Item>
{
  friend class ItemEnumeratorCS;
  friend class ItemVectorView;
  // NOTE: Normalement il suffirait de faire cela:
  //   template<class T> friend class ItemEnumeratorBase;
  // mais cela ne fonctionne pas avec GCC 8. On fait donc la spécialisation
  // à la main
  friend class ItemEnumeratorBaseV3T<Node>;
  friend class ItemEnumeratorBaseV3T<ItemWithNodes>;
  friend class ItemEnumeratorBaseV3T<Edge>;
  friend class ItemEnumeratorBaseV3T<Face>;
  friend class ItemEnumeratorBaseV3T<Cell>;
  friend class ItemEnumeratorBaseV3T<Particle>;
  friend class ItemEnumeratorBaseV3T<DoF>;

 public:

  typedef ItemInternal* ItemInternalPtr;
  using BaseClass = ItemEnumeratorBaseT<Item>;

 public:

  ItemEnumerator() = default;
  ItemEnumerator(const ItemInternalVectorView& view) : BaseClass(view){}
  ItemEnumerator(const ItemInternalEnumerator& rhs) : BaseClass(rhs,true){}

  // TODO: make deprecated
  ItemEnumerator(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup){}
  // TODO: make deprecated
  ItemEnumerator(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}
  // TODO: make deprecated
  ItemEnumerator(const ItemInternalVectorView& view, const ItemGroupImpl* agroup)
  : BaseClass(view,agroup){}

 protected:

  ItemEnumerator(ItemSharedInfo* s,const Int32ConstArrayView& local_ids)
  : BaseClass(s,local_ids){}

 public:

  static ItemEnumerator fromItemEnumerator(const ItemEnumerator& rhs)
  {
    return ItemEnumerator(rhs);
  }

 private:

  ItemEnumerator(ItemSharedInfo* shared_info,const Int32* local_ids,Int32 index,Int32 n,
                 const ItemGroupImpl* agroup,impl::ItemBase item_base)
  : BaseClass(shared_info,local_ids,index,n,agroup,item_base){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructeur seulement utilisé par fromItemEnumerator()
inline ItemEnumeratorBaseV3::
ItemEnumeratorBaseV3(const ItemEnumerator& rhs,bool)
: m_shared_info(rhs.m_shared_info)
, m_local_ids(rhs.unguardedLocalIds())
, m_index(rhs.index())
, m_count(rhs.count())
, m_group_impl(rhs.group())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructeur seulement utilisé par fromItemEnumerator()
template<typename ItemType> inline ItemEnumeratorBaseV3T<ItemType>::
ItemEnumeratorBaseV3T(const ItemEnumerator& rhs,bool v)
: ItemEnumeratorBaseV3(rhs,v)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemEnumeratorBaseV3::
ItemEnumeratorBaseV3(const ItemEnumerator& rhs)
: m_shared_info(rhs.m_shared_info)
, m_local_ids(rhs.unguardedLocalIds())
, m_index(rhs.index())
, m_count(rhs.count())
, m_group_impl(rhs.group())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemEnumeratorBaseV3T<ItemType>::
ItemEnumeratorBaseV3T(const ItemEnumerator& rhs)
: ItemEnumeratorBaseV3(rhs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemEnumeratorBaseV3T<ItemType>::
ItemEnumeratorBaseV3T(const ItemInternalEnumerator& rhs)
: ItemEnumeratorBaseV3T(ItemEnumerator(rhs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemEnumerator ItemEnumeratorBaseV3T<ItemType>::
toItemEnumerator() const
{
  return ItemEnumerator(m_shared_info,m_local_ids,m_index,m_count,m_group_impl,m_item_for_operator_arrow);
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
  friend class ItemVectorViewT<ItemType>;

 public:

  ItemEnumeratorT() = default;
  ItemEnumeratorT(const ItemVectorView& rhs) : BaseClass(rhs){}
  ItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs) : BaseClass(rhs){}

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemEnumerator& rhs)
  : BaseClass(rhs){}

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemInternalEnumerator& rhs)
  : BaseClass(rhs){}

  // TODO: rendre obsolète
  ItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup){}
  // TODO: rendre obsolète
  ItemEnumeratorT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}
  // TODO: rendre obsolète
  ItemEnumeratorT(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view,agroup){}

 private:

  ItemEnumeratorT(ItemSharedInfo* s,const Int32ConstArrayView& local_ids)
  : BaseClass(s,local_ids){}

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
  return ItemEnumerator(m_shared_info,m_local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId::
ItemLocalId(ItemEnumerator enumerator)
: m_local_id(enumerator.asItemLocalId())
{
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

// TODO: ajouter vérification du bon type
template<typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemEnumerator enumerator)
: ItemLocalId(enumerator.asItemLocalId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemEnumeratorT<ItemType> enumerator)
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

//! Enumérateur generique d'un groupe de noeuds duals
#define ENUMERATE_DUALNODE(name,group) ENUMERATE_(::Arcane::DualNode,name,group)

//! Enumérateur generique d'un groupe de liaisons
#define ENUMERATE_LINK(name,group) ENUMERATE_(::Arcane::Link,name,group)

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
