// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumerator.h                                            (C) 2000-2021 */
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
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;
  friend class ItemEnumeratorCS;

 public:

  ItemEnumerator()
  : m_items(0), m_local_ids(0), m_index(0), m_count(0), m_group_impl(0) {}
  ItemEnumerator(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = 0)
  : m_items(items), m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) {}
  ItemEnumerator(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = 0)
  : m_items(items.data()), m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) {}
  ItemEnumerator(const ItemInternalVectorView& view, const ItemGroupImpl * agroup = 0)
  : m_items(view.items().data()), m_local_ids(view.localIds().data()),
    m_index(0), m_count(view.size()), m_group_impl(agroup) {}
  ItemEnumerator(const ItemEnumerator& rhs)
  : m_items(rhs.m_items), m_local_ids(rhs.m_local_ids),
    m_index(rhs.m_index), m_count(rhs.m_count), m_group_impl(rhs.m_group_impl) {}
  ItemEnumerator(const ItemInternalEnumerator& rhs)
  : m_items(rhs.m_items), m_local_ids(rhs.m_local_ids),
    m_index(rhs.m_index), m_count(rhs.m_count), m_group_impl(0) {}

 public:

  Item operator*() const { return m_items[ m_local_ids[m_index] ]; }
  Item operator->() const { return m_items[ m_local_ids[m_index] ]; }

  constexpr void operator++() { ++m_index; }
  constexpr bool operator()() { return m_index<m_count; }
  constexpr bool hasNext() { return m_index<m_count; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr Int32 itemLocalId() const { return m_local_ids[m_index]; }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_local_ids[m_index]; }

  //! Indices locaux
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  //! Indices locaux
  constexpr const ItemInternalPtr* unguardedItems() const { return m_items; }

  //! Partie interne (pour usage interne uniquement)
  constexpr ItemInternal* internal() const { return m_items[m_local_ids[m_index]]; }

  //! Groupe sous-jacent s'il existe (0 sinon)
  /*! Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites */
  constexpr const ItemGroupImpl * group() const { return m_group_impl; }

  constexpr ItemLocalId asItemLocalId() const { return ItemLocalId{m_local_ids[m_index]}; }

  //! Conversion vers un 'ItemLocalId'
  constexpr operator ItemLocalId() const
  {
    return ItemLocalId(m_local_ids[m_index]);
  }

 protected:

  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;
  Integer m_count;
  const ItemGroupImpl * m_group_impl; // pourrait être retiré en mode release si nécessaire
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste typée d'entités
 */
template<typename ItemType>
class ItemEnumeratorT
: public ItemEnumerator
{
 public:

  typedef typename ItemType::LocalIdType LocalIdType;

 public:

  ItemEnumeratorT() {}
  ItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = 0)
    : ItemEnumerator(items,local_ids,n,agroup) {}
  ItemEnumeratorT(const ItemInternalEnumerator& rhs)
  : ItemEnumerator(rhs) {}
  ItemEnumeratorT(const ItemEnumerator& rhs)
  : ItemEnumerator(rhs) {}
  ItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumerator(rhs) {}

 public:

  ItemType operator*() const
  {
    return ItemType(m_items,m_local_ids[m_index]);
  }
  ItemType operator->() const
  {
    return ItemType(m_items[m_local_ids[m_index]]);
  }

  constexpr LocalIdType asItemLocalId() const { return LocalIdType{m_local_ids[m_index]}; }

  operator LocalIdType() const
  {
    return LocalIdType(m_local_ids[m_index]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemEnumerator ItemVectorView::
enumerator() const
{
  return ItemEnumerator(m_items.data(),m_local_ids.localIds().data(),
                        m_local_ids.size());
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
  for( A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) iname((view).enumerator() A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumérateur générique d'un groupe d'entité
#define ENUMERATE_(type,name,group) A_ENUMERATE_ITEM(ItemEnumeratorT< type >,name,group)

//! Enumérateur générique d'un groupe d'entité
#define ENUMERATE_GENERIC(type,name,group) A_ENUMERATE_ITEM(ItemEnumeratorT< type >,name,group)

//! Enumérateur générique d'un groupe de noeuds
#define ENUMERATE_ITEM(name,group) A_ENUMERATE_ITEM(ItemEnumerator,name,group)

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
