// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumeratorBase.h                                        (C) 2000-2022 */
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
  friend class SimdItemEnumeratorBase;

 protected:

  ItemEnumeratorBase()
  : m_local_ids(nullptr), m_index(0), m_count(0), m_group_impl(nullptr) { }
  ItemEnumeratorBase(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = nullptr)
  : m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) { _init(items); }
  ItemEnumeratorBase(ItemSharedInfo* shared_info,const Int32ConstArrayView& local_ids)
  : m_shared_info(shared_info), m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(nullptr) { }
  ItemEnumeratorBase(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) { _init(items.data()); }
  ItemEnumeratorBase(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : m_shared_info(view.m_shared_info), m_local_ids(view.localIds().data()),
    m_index(0), m_count(view.size()), m_group_impl(agroup) { }
  ItemEnumeratorBase(const ItemVectorView& rhs)
  : ItemEnumeratorBase((const ItemInternalVectorView&)rhs) {}

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
  constexpr Int32 itemLocalId() const { return m_local_ids[m_index]; }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_local_ids[m_index]; }

  /*!
   * \internal
   * \brief Indices locaux.
   */
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  /*!
   * \internal
   * \brief Liste des ItemInternal.
   */
  constexpr const ItemInternalPtr* unguardedItems() const { return m_shared_info->m_items_internal.data(); }

  /*!
   * \internal
   * \brief Partie interne (pour usage interne uniquement).
   */
  constexpr ItemInternal* internal() const { return m_shared_info->m_items_internal[m_local_ids[m_index]]; }

  /*!
   * \brief Groupe sous-jacent s'il existe (nullptr sinon)
   *
   * \brief Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites.
   */
  constexpr const ItemGroupImpl* group() const { return m_group_impl; }

  static constexpr int version() { return 3; }

 protected:

  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
  const Int32* ARCANE_RESTRICT m_local_ids;
  Int32 m_index;
  Int32 m_count;
  const ItemGroupImpl* m_group_impl; // pourrait être retiré en mode release si nécessaire

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBase(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBase(ItemSharedInfo* shared_info,const Int32* local_ids,Int32 index,Int32 n,
                       const ItemGroupImpl * agroup)
  : m_shared_info(shared_info), m_local_ids(local_ids), m_index(index), m_count(n), m_group_impl(agroup)
  {
  }

 private:

  void _init(const ItemInternalPtr* items)
  {
    m_shared_info = (items) ? items[0]->sharedInfo() : ItemSharedInfo::nullInstance();
  }
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
 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;
  using BaseClass = ItemEnumeratorBase;

 protected:

  ItemEnumeratorBaseT()
  : BaseClass() { }
  ItemEnumeratorBaseT(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup) {}
  ItemEnumeratorBaseT(ItemSharedInfo* shared_info,const Int32ConstArrayView& local_ids)
  : BaseClass(shared_info,local_ids){}
  ItemEnumeratorBaseT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}
  ItemEnumeratorBaseT(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view,agroup) {}
  ItemEnumeratorBaseT(const ItemVectorView& rhs)
  : ItemEnumeratorBaseT((const ItemInternalVectorView&)rhs) {}
  ItemEnumeratorBaseT(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumeratorBaseT((const ItemInternalVectorView&)rhs) {}

  ItemEnumeratorBaseT(const ItemEnumerator& rhs);
  ItemEnumeratorBaseT(const ItemInternalEnumerator& rhs);

 public:

  constexpr ItemType operator*() const { return ItemType(m_local_ids[m_index],m_shared_info); }
  constexpr const ItemType* operator->() const { m_item_for_operator_arrow = this->operator*(); return &m_item_for_operator_arrow; }

  constexpr LocalIdType asItemLocalId() const { return LocalIdType{m_local_ids[m_index]}; }

  ItemEnumerator toItemEnumerator() const;

 protected:

  mutable ItemType m_item_for_operator_arrow = ItemType(NULL_ITEM_LOCAL_ID,nullptr);

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseT(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseT(ItemSharedInfo* shared_info,const Int32* local_ids,Int32 index,Int32 n,
                        const ItemGroupImpl* agroup,Item item_base)
  : ItemEnumeratorBase(shared_info,local_ids,index,n,agroup), m_item_for_operator_arrow(item_base)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
