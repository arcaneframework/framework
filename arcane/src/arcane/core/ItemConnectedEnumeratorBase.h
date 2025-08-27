// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectedEnumeratorBase.h                               (C) 2000-2025 */
/*                                                                           */
/* Classe de base des énumérateurs sur les entités connectées du maillage.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMCONNECTEDENUMERATORBASE_H
#define ARCANE_CORE_ITEMCONNECTEDENUMERATORBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternalEnumerator.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe de base des énumérateurs sur une liste d'entité connectées.
 *
 * Les instances de cette classes sont créées soit via ItemConnectedEnumerator,
 * soit via ItemConnectedEnumeratorT.
 *
 * \code
 * for( ItemConnectedEnumeratorBase iter(...); iter.hasNext(); ++iter )
 *   ;
 * \endcode
 */
class ItemConnectedEnumeratorBase
{
  // Seule ces classes ont le droit de construire des instances de cette classe
  template <typename T> friend class ItemConnectedEnumeratorBaseT;

 private:

  ItemConnectedEnumeratorBase() = default;
  explicit ItemConnectedEnumeratorBase(const ConstArrayView<Int32> local_ids)
  : m_local_ids(local_ids.data())
  , m_count(local_ids.size())
  {}
  template <int E> explicit ItemConnectedEnumeratorBase(const ItemConnectedListView<E>& rhs)
  : m_local_ids(rhs._localIds().data())
  , m_count(rhs._localIds().size())
  , m_local_id_offset(rhs._localIdOffset())
  {}
  ItemConnectedEnumeratorBase(const Int32* local_ids, Int32 index, Int32 n)
  : m_local_ids(local_ids)
  , m_index(index)
  , m_count(n)
  {
  }

 public:

  //! Incrémente l'index de l'énumérateur
  constexpr void operator++()
  {
    ++m_index;
  }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool operator()() const
  {
    return m_index < m_count;
  }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool hasNext() const { return m_index < m_count; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Int32 count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Int32 index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr ItemLocalId itemLocalId() const { return ItemLocalId(m_local_id_offset + m_local_ids[m_index]); }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_local_id_offset + m_local_ids[m_index]; }

 protected:

  const Int32* ARCANE_RESTRICT m_local_ids = nullptr;
  Int32 m_index = 0;
  Int32 m_count = 0;
  Int32 m_local_id_offset = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base typeé des énumérateurs sur une liste d'entité connectées.
 *
 * Les instances de cette classes sont créées soit via ItemConnectedEnumerator, soit
 * via ItemConnectedEnumeratorT.
 */
template <typename ItemType>
class ItemConnectedEnumeratorBaseT
: public ItemConnectedEnumeratorBase
{
  friend class ItemConnectedEnumerator;
  friend class ItemConnectedEnumeratorT<ItemType>;

 private:

  using LocalIdType = typename ItemType::LocalIdType;
  using BaseClass = ItemConnectedEnumeratorBase;

 private:

  ItemConnectedEnumeratorBaseT()
  : BaseClass()
  , m_item(NULL_ITEM_LOCAL_ID, ItemSharedInfo::nullInstance())
  {}

  ItemConnectedEnumeratorBaseT(ItemSharedInfo* shared_info, const Int32ConstArrayView& local_ids)
  : BaseClass(local_ids)
  , m_item(NULL_ITEM_LOCAL_ID, shared_info)
  {}

  ItemConnectedEnumeratorBaseT(const impl::ItemIndexedListView<DynExtent>& view)
  : ItemConnectedEnumeratorBaseT(view.m_shared_info, view.constLocalIds())
  {}

  ItemConnectedEnumeratorBaseT(const ItemConnectedListViewT<ItemType>& rhs)
  : BaseClass(rhs)
  , m_item(NULL_ITEM_LOCAL_ID, rhs.m_shared_info)
  {}

  ItemConnectedEnumeratorBaseT(const Int32* local_ids, Int32 index, Int32 n, Item item_base)
  : ItemConnectedEnumeratorBase(local_ids, index, n)
  , m_item(item_base)
  {
  }

 public:

  constexpr ItemType operator*() const
  {
    m_item.m_local_id = m_local_id_offset + m_local_ids[m_index];
    return m_item;
  }
  constexpr const ItemType* operator->() const
  {
    m_item.m_local_id = m_local_id_offset + m_local_ids[m_index];
    return &m_item;
  }

  constexpr LocalIdType asItemLocalId() const
  {
    return LocalIdType{ m_local_id_offset + m_local_ids[m_index] };
  }

 protected:

  mutable ItemType m_item = ItemType(NULL_ITEM_LOCAL_ID, nullptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
