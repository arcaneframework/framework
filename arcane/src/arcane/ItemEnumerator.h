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
 * \brief Classe de base des énumérateurs sur une liste d'entité.
 *
 * Les instances de cette classes sont créées soit via ItemEnumerator, soit
 * via ItemEnumeratorT.
 */
template<typename ItemType>
class ItemEnumeratorBaseT
{
 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;

 protected:

  ItemEnumeratorBaseT()
  : m_items(nullptr), m_local_ids(nullptr), m_index(0), m_count(0), m_group_impl(nullptr) { _init(); }
  ItemEnumeratorBaseT(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = nullptr)
  : m_items(items), m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : m_items(items.data()), m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseT(const ItemInternalVectorView& view, const ItemGroupImpl * agroup = nullptr)
  : m_items(view.items().data()), m_local_ids(view.localIds().data()),
    m_index(0), m_count(view.size()), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseT(const ItemVectorView& rhs)
  : ItemEnumeratorBaseT((const ItemInternalVectorView&)rhs) {}
  ItemEnumeratorBaseT(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumeratorBaseT((const ItemInternalVectorView&)rhs) {}

  ItemEnumeratorBaseT(const ItemEnumerator& rhs);
  ItemEnumeratorBaseT(const ItemInternalEnumerator& rhs);

 public:

  //! Incrémente l'index de l'énumérateur
  constexpr void operator++()
  {
    ++m_index;
    m_is_not_end = (m_index<m_count);
    if (m_is_not_end)
      m_base.m_local_id = m_local_ids[m_index];
  }
  constexpr bool operator()() { return m_is_not_end; }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool hasNext() { return m_is_not_end; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr Int32 itemLocalId() const { return m_base.m_local_id; }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_base.m_local_id; }

  /*!
   * \internal
   * \brief Indices locaux.
   */
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  /*!
   * \internal
   * \brief Liste des ItemInternal.
   */
  constexpr const ItemInternalPtr* unguardedItems() const { return m_items; }

  /*!
   * \internal
   * \brief Partie interne (pour usage interne uniquement).
   */
  constexpr ItemInternal* internal() const { return m_items[m_base.m_local_id]; }

  /*!
   * \brief Groupe sous-jacent s'il existe (nullptr sinon)
   *
   * \brief Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites.
   */
  constexpr const ItemGroupImpl* group() const { return m_group_impl; }

  constexpr ItemType operator*() const { return m_base; }
  constexpr const ItemType* operator->() const { return &m_base; }

  constexpr LocalIdType asItemLocalId() const { return LocalIdType{m_base.m_local_id}; }

 public:

  ItemEnumerator toItemEnumerator() const;

 protected:

  // TODO Rendre privé
  ItemType m_base;
  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Int32 m_index;
  Int32 m_count;
  bool m_is_not_end;
  const ItemGroupImpl* m_group_impl; // pourrait être retiré en mode release si nécessaire

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseT(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseT(const ItemInternalPtr* items,const Int32* local_ids,Int32 index,Int32 n,
                      const ItemGroupImpl * agroup,ItemBase item_base)
  : m_base(item_base), m_items(items), m_local_ids(local_ids), m_index(index), m_count(n), m_group_impl(agroup)
  {
    m_is_not_end = (m_index<m_count);
  }

 private:

  void _init()
  {
    m_is_not_end = (m_index<m_count);
    if (m_is_not_end){
      Int32 lid = m_local_ids[m_index];
      // Vérifie qu'on n'indexe pas 'm_items' avec un localId() nul.
      // Cela n'est pas possible avec les groupes d'entités mais ca l'est avec
      // les ItemVector par exemple.
      // De même, 'm_items' peut-être nul si tous les localId() sont nuls.
      // Si c'est le cas, alors on prend comme ItemSharedInfo l'instance nulle.
      // A terme le ItemSharedInfo sera renseigné directement dans le constructeur de
      // cette classe.
      Int32 idx = lid;
      if (idx==NULL_ITEM_LOCAL_ID)
        idx = 0;
      ItemSharedInfo* isi = (m_items) ? m_items[idx]->sharedInfo() : ItemSharedInfo::nullInstance();
      m_base = ItemType(ItemBaseBuildInfo(lid,isi));
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste d'entités.
 */
class ItemEnumerator
: public ItemEnumeratorBaseT<Item>
{
  friend class ItemEnumeratorCS;
  template<class T> friend class ItemEnumeratorBaseT;

 public:

  typedef ItemInternal* ItemInternalPtr;
  using BaseClass = ItemEnumeratorBaseT<Item>;

 public:

  ItemEnumerator() = default;
  ItemEnumerator(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup){}
  ItemEnumerator(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}
  ItemEnumerator(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view,agroup){}
  ItemEnumerator(const ItemInternalEnumerator& rhs)
  : BaseClass(rhs,true){}

 public:

  static ItemEnumerator fromItemEnumerator(const ItemEnumerator& rhs)
  {
    return ItemEnumerator(rhs);
  }

 private:

  ItemEnumerator(const ItemInternalPtr* items,const Int32* local_ids,Int32 index,Int32 n,
                 const ItemGroupImpl* agroup,ItemBase item_base)
  : BaseClass(items,local_ids,index,n,agroup,item_base){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructeur seulement utilisé par fromItemEnumerator()
template<typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemEnumerator& rhs,bool)
: m_items(rhs.unguardedItems())
, m_local_ids(rhs.unguardedLocalIds())
, m_index(rhs.index())
, m_count(rhs.count())
, m_group_impl(rhs.group())
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemEnumeratorBaseT<ItemType>::
ItemEnumeratorBaseT(const ItemEnumerator& rhs)
: m_items(rhs.unguardedItems())
, m_local_ids(rhs.unguardedLocalIds())
, m_index(rhs.index())
, m_count(rhs.count())
, m_group_impl(rhs.group())
{
  _init();
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
  return ItemEnumerator(m_items,m_local_ids,m_index,m_count,m_group_impl,m_base);
}

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

 public:

  ItemEnumeratorT()
  : BaseClass() {}
  ItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup){}
  ItemEnumeratorT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}
  ItemEnumeratorT(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view,agroup){}
  ItemEnumeratorT(const ItemVectorView& rhs)
  : BaseClass(rhs){}
  ItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs)
  : BaseClass(rhs){}

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemEnumerator& rhs)
  : BaseClass(rhs){}

  [[deprecated("Y2021: Use strongly typed enumerator (Node, Face, Cell, ...) instead of generic (Item) enumerator")]]
  ItemEnumeratorT(const ItemInternalEnumerator& rhs)
  : BaseClass(rhs){}

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
  return ItemEnumerator(m_items.data(),m_local_ids.localIds().data(),
                        m_local_ids.size());
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
