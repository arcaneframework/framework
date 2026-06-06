// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectedEnumerator.h                                   (C) 2000-2023 */
/*                                                                           */
/* Enumerators for connected entities of the mesh.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMCONNECTEDENUMERATOR_H
#define ARCANE_ITEMCONNECTEDENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"
#include "arcane/core/ItemConnectedEnumeratorBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ItemConnectedEnumerator.h
 *
 * \brief Types and macros for iterating over mesh entities connected
 * to another entity.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a list of entities connected to another.
 */
class ItemConnectedEnumerator
: public ItemConnectedEnumeratorBaseT<Item>
{
  friend class ItemGroup;
  friend class ItemVector;
  friend class ItemVectorView;
  friend class ItemPairEnumerator;
  template <int Extent> friend class ItemConnectedListView;
  // NOTE: Normally, it would suffice to do this:
  //   template<class T> friend class ItemConnectedEnumeratorBase;
  // but this does not work with GCC 8. So we do the specialization
  // manually
  friend class ItemConnectedEnumeratorBaseT<Item>;
  friend class ItemConnectedEnumeratorBaseT<Node>;
  friend class ItemConnectedEnumeratorBaseT<ItemWithNodes>;
  friend class ItemConnectedEnumeratorBaseT<Edge>;
  friend class ItemConnectedEnumeratorBaseT<Face>;
  friend class ItemConnectedEnumeratorBaseT<Cell>;
  friend class ItemConnectedEnumeratorBaseT<Particle>;
  friend class ItemConnectedEnumeratorBaseT<DoF>;

 public:

  using BaseClass = ItemConnectedEnumeratorBaseT<Item>;

 public:

  ItemConnectedEnumerator() = default;

 public:

  ItemConnectedEnumerator(const impl::ItemIndexedListView<DynExtent>& rhs)
  : BaseClass(rhs)
  {}

  template <int E> ItemConnectedEnumerator(const ItemConnectedListView<E>& rhs)
  : BaseClass(ItemConnectedListViewT<Item, E>(rhs))
  {}

 protected:

  ItemConnectedEnumerator(ItemSharedInfo* s, const Int32ConstArrayView& local_ids)
  : BaseClass(s, local_ids)
  {}

 private:

  ItemConnectedEnumerator(const Int32* local_ids, Int32 index, Int32 n, Item item_base)
  : BaseClass(local_ids, index, n, item_base)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a typed list of connected entities of type \a ItemType
 */
template <typename ItemType>
class ItemConnectedEnumeratorT
: public ItemConnectedEnumeratorBaseT<ItemType>
{
  using BaseClass = ItemConnectedEnumeratorBaseT<ItemType>;

 public:

  ItemConnectedEnumeratorT() = default;
  template <int E> ItemConnectedEnumeratorT(const ItemConnectedListView<E>& rhs)
  : BaseClass(rhs)
  {}
  ItemConnectedEnumeratorT(const ItemConnectedListViewT<ItemType>& rhs)
  : BaseClass(rhs)
  {}

 private:

  ItemConnectedEnumeratorT(ItemSharedInfo* s, const Int32ConstArrayView& local_ids)
  : BaseClass(s, local_ids)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \def ENUMERATE_CONNECTED_(type,iterator_name,item,connectivity_func)
 *
 * \brief Macro to iterate over a list of entities connected to another entity.
 *
 * \warning Experimental API. Do not use outside of %Arcane.
 *
 * \param type type of the connected entity (Node, Face, Cell, Edge, Particle, DoF )
 * \param iterator_name name of the enumerator
 * \param item name of the entity whose connectivities are desired
 * \param connectivity_func method of \a item to retrieve the connectivity.
 *
 * Example for iterating over the mesh nodes:
 * \code
 * Arcane::Cell cell = ...;
 * ENUMERATE_CONNECTED_(Node,inode,cell,nodes()){
 *   Arcane::Node node(*inode);
 *   info() << "Node local_id=" << node.localId()
 * }
 * \endcode
 */
#ifdef ARCANE_USE_SPECIFIC_ITEMCONNECTED

#define ENUMERATE_CONNECTED_(type, iterator_name, item, connectivity_func) \
  for (::Arcane::ItemConnectedEnumeratorT<type> iterator_name((item).connectivity_func); iterator_name.hasNext(); ++iterator_name)

#else

#define ENUMERATE_CONNECTED_(type, iterator_name, item, connectivity_func) \
  for (::Arcane::ItemEnumeratorT<type> iterator_name((item).connectivity_func); iterator_name.hasNext(); ++iterator_name)

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
