// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectedEnumerator.h                                   (C) 2000-2023 */
/*                                                                           */
/* Enumérateurs sur les entités connectées du maillage.                      */
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
 * \brief Types et macros pour itérer sur les entités du maillage connectées
 * à une autre entité.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste d'entités connectées à une autre.
 */
class ItemConnectedEnumerator
: public ItemConnectedEnumeratorBaseT<Item>
{
  friend class ItemGroup;
  friend class ItemVector;
  friend class ItemVectorView;
  friend class ItemPairEnumerator;
  template<int Extent> friend class ItemConnectedListView;
  // NOTE: Normalement il suffirait de faire cela:
  //   template<class T> friend class ItemConnectedEnumeratorBase;
  // mais cela ne fonctionne pas avec GCC 8. On fait donc la spécialisation
  // à la main
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

  template<int E> ItemConnectedEnumerator(const ItemConnectedListView<E>& rhs)
  : BaseClass(ItemConnectedListViewT<Item,E>(rhs)){}

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
 * \brief Enumérateur sur une liste typée d'entités connectées de type \a ItemType
 */
template<typename ItemType>
class ItemConnectedEnumeratorT
: public ItemConnectedEnumeratorBaseT<ItemType>
{
  using BaseClass = ItemConnectedEnumeratorBaseT<ItemType>;

 public:

  ItemConnectedEnumeratorT() = default;
  template<int E> ItemConnectedEnumeratorT(const ItemConnectedListView<E>& rhs) : BaseClass(rhs){}
  ItemConnectedEnumeratorT(const ItemConnectedListViewT<ItemType>& rhs) : BaseClass(rhs){}

 private:

  ItemConnectedEnumeratorT(ItemSharedInfo* s,const Int32ConstArrayView& local_ids)
  : BaseClass(s,local_ids){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \def ENUMERATE_CONNECTED_(type,iterator_name,item,connectivity_func)
 *
 * \brief Macro pour itérer sur une liste d'entité connectées à une autre entité.
 *
 * \warning API expérimentale. Ne pas utiliser en dehors de %Arcane.
 *
 * \param type type de l'entité connectée (Node, Face, Cell, Edge, Particle, DoF )
 * \param iterator_name nom de l'énumérateur
 * \param item nom de l'entité dont on souhaite avoir les connectivités
 * \param connectivity_func méthode de \a item pour récupérer la connectivité.
 *
 * Exemple pour itérer sur les noeuds de la mailles:
 * \code
 * Arcane::Cell cell = ...;
 * ENUMERATE_CONNECTED_(Node,inode,cell,nodes()){
 *   Arcane::Node node(*inode);
 *   info() << "Node local_id=" << node.localId()
 * }
 * \endcode
 */
#ifdef ARCANE_USE_SPECIFIC_ITEMCONNECTED

#define ENUMERATE_CONNECTED_(type,iterator_name,item,connectivity_func) \
  for( ::Arcane::ItemConnectedEnumeratorT< type > iterator_name( (item) . connectivity_func ); iterator_name . hasNext(); ++iterator_name )

#else

#define ENUMERATE_CONNECTED_(type,iterator_name,item,connectivity_func) \
  for( ::Arcane::ItemEnumeratorT< type > iterator_name( (item) . connectivity_func ); iterator_name . hasNext(); ++iterator_name )

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
