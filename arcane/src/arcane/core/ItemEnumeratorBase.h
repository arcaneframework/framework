// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumeratorBase.h                                        (C) 2000-2023 */
/*                                                                           */
/* Classe de base des énumérateurs sur les entités du maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMENUMERATORBASE_H
#define ARCANE_ITEMENUMERATORBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternalEnumerator.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
#define ARCANE_LOCALID_ADD_OFFSET(a) (m_local_id_offset + (a))
#else
#define ARCANE_LOCALID_ADD_OFFSET(a) (a)
#endif

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
class ItemEnumeratorBase
{
 private:

  using ItemInternalPtr = ItemInternal*;

 protected:

  ItemEnumeratorBase()
  : m_local_ids(nullptr), m_index(0), m_count(0), m_group_impl(nullptr) { }
  ItemEnumeratorBase(const ItemInternalPtr*,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup)
  : m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) { }
  explicit ItemEnumeratorBase(const Int32ConstArrayView& local_ids)
  : m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(nullptr) { }
  ItemEnumeratorBase(const Int32ConstArrayView& local_ids,const ItemGroupImpl* agroup)
  : m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) { }
  ItemEnumeratorBase(const ItemInternalVectorView& view,const ItemGroupImpl* agroup)
  : m_local_ids(view.localIds().data()), m_index(0), m_count(view.size()), m_group_impl(agroup) { }
  ItemEnumeratorBase(const ItemVectorView& rhs)
  : ItemEnumeratorBase((const ItemInternalVectorView&)rhs,nullptr) {}
  template<int E> ItemEnumeratorBase(const ItemConnectedListView<E>& rhs)
  : m_local_ids(rhs._localIds().data()), m_count(rhs._localIds().size()){}

  ItemEnumeratorBase(const ItemEnumerator& rhs);
  ItemEnumeratorBase(const ItemInternalEnumerator& rhs);

 public:

  //! Incrémente l'index de l'énumérateur
  constexpr void operator++() { ++m_index; }
  constexpr bool operator()() { return m_index<m_count; }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool hasNext() { return m_index<m_count; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr Int32 itemLocalId() const { return ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index]); }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index]); }

  /*!
   * \internal
   * \brief Indices locaux.
   */
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  /*!
   * \brief Groupe sous-jacent s'il existe (nullptr sinon)
   *
   * \brief Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites.
   */
  constexpr const ItemGroupImpl* group() const { return m_group_impl; }

  static constexpr int version() { return 3; }

 protected:

  const Int32* ARCANE_RESTRICT m_local_ids;
  Int32 m_index = 0;
  Int32 m_count;
#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
  Int32 m_local_id_offset = 0;
#endif
  const ItemGroupImpl* m_group_impl = nullptr; // pourrait être retiré en mode release si nécessaire

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBase(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBase(const Int32* local_ids,Int32 index,Int32 n, const ItemGroupImpl * agroup)
  : m_local_ids(local_ids), m_index(index), m_count(n), m_group_impl(agroup)
  {
  }

  constexpr ItemInternal* _internal(ItemSharedInfo* si) const
  {
    return si->m_items_internal[ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index])];
  }
  constexpr const ItemInternalPtr* _unguardedItems(ItemSharedInfo* si) const
  {
    return si->m_items_internal.data();
  }

 private:
};

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
: public ItemEnumeratorBase
{
  friend class SimdItemEnumeratorBase;

 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;
  using BaseClass = ItemEnumeratorBase;

 protected:

  ItemEnumeratorBaseT()
  : BaseClass(), m_item(NULL_ITEM_LOCAL_ID,ItemSharedInfo::nullInstance()){}
  ItemEnumeratorBaseT(ItemSharedInfo* shared_info,const Int32ConstArrayView& local_ids)
  : BaseClass(local_ids), m_item(NULL_ITEM_LOCAL_ID,shared_info) {}
  ItemEnumeratorBaseT(const ItemInfoListView& items,const Int32ConstArrayView& local_ids,const ItemGroupImpl* agroup)
  : BaseClass(local_ids,agroup), m_item(NULL_ITEM_LOCAL_ID,items.m_item_shared_info){ }
  ItemEnumeratorBaseT(const ItemInternalVectorView& view,const ItemGroupImpl* agroup)
  : BaseClass(view,agroup), m_item(NULL_ITEM_LOCAL_ID,view.m_shared_info) { }
  ItemEnumeratorBaseT(const ItemVectorView& rhs)
  : ItemEnumeratorBaseT((const ItemInternalVectorView&)rhs,nullptr) {}
  ItemEnumeratorBaseT(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumeratorBaseT((const ItemInternalVectorView&)rhs,nullptr) {}

  ItemEnumeratorBaseT(const ItemEnumerator& rhs);
  ItemEnumeratorBaseT(const impl::ItemIndexedListView<DynExtent>& view)
  : ItemEnumeratorBaseT(view.m_shared_info, view.constLocalIds()){}

  ItemEnumeratorBaseT(const ItemConnectedListViewT<ItemType>& rhs)
  : BaseClass(rhs), m_item(NULL_ITEM_LOCAL_ID,rhs.m_shared_info){}

 protected:

  // TODO: a supprimer
  ItemEnumeratorBaseT(const ItemInternalPtr* items,const Int32* local_ids,Integer n,const ItemGroupImpl* agroup)
  : BaseClass(items,local_ids,n,agroup) { _init(items); }
  // TODO: a supprimer
  ItemEnumeratorBaseT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids,const ItemGroupImpl* agroup)
  : BaseClass(local_ids,agroup){ _init(items.data()); }
  // TODO: a supprimer
  ItemEnumeratorBaseT(const ItemInternalEnumerator& rhs);

 public:

  /*!
   * \internal
   * \brief Liste des ItemInternal.
   * NOTE: Dans Arcane, méthode utilisée uniquement pour le wrapper C#. A supprimer ensuite
   */
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane")
  constexpr const ItemInternalPtr* unguardedItems() const { return _unguardedItems(m_item.m_shared_info); }

  /*!
   * \internal
   * \brief Partie interne (pour usage interne uniquement).
   */
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane")
  constexpr ItemInternal* internal() const { return _internal(m_item.m_shared_info); }

 public:

  constexpr ItemType operator*() const
  {
    m_item.m_local_id = ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index]);
    return m_item;
  }
  constexpr const ItemType* operator->() const
  {
    m_item.m_local_id = ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index]);
    return &m_item;
  }

  constexpr LocalIdType asItemLocalId() const
  {
    return LocalIdType{ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index])};
  }

  constexpr operator LocalIdType() const
  {
    return LocalIdType{ARCANE_LOCALID_ADD_OFFSET(m_local_ids[m_index])};
  }

  ItemEnumerator toItemEnumerator() const;

 public:

  impl::ItemBase _internalItemBase() const { return m_item.itemBase(); }

 protected:

  mutable ItemType m_item = ItemType(NULL_ITEM_LOCAL_ID,nullptr);

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseT(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseT(const Int32* local_ids,Int32 index,Int32 n,
                      const ItemGroupImpl* agroup,Item item_base)
  : ItemEnumeratorBase(local_ids,index,n,agroup), m_item(item_base)
  {
  }

  void _init(const ItemInternalPtr* items)
  {
    m_item.m_shared_info = ItemInternalCompatibility::_getSharedInfo(items,count());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

#undef ARCANE_LOCALID_ADD_OFFSET

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
