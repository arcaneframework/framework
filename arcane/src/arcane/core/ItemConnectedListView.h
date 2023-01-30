// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectedListView.h                                     (C) 2000-2023 */
/*                                                                           */
/* Vue sur une liste d'entités connectés à une autre entité.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMCONNECTEDLISTVIEW_H
#define ARCANE_ITEMCONNECTEDLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternalVectorView.h"
#include "arcane/ItemIndexArrayView.h"
#include "arcane/ItemInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un vecteur d'entités.
 *
 * \warning la vue n'est valide que tant que le tableau associé n'est
 * pas modifié et que la famille d'entité associée à ce tableau n'est
 * elle même pas modifiée.
 */
template<int Extent>
class ItemConnectedListView
{
  friend ItemVector;
  friend class ItemEnumeratorBase;
  template<typename ItemType> friend class ItemEnumeratorBaseT;

 public:

  using const_iterator = ItemVectorViewConstIterator;
  using difference_type = std::ptrdiff_t;
  using value_type = Item;
  using reference_type = Item&;
  using const_reference_type = const Item&;

 public:

  ItemConnectedListView() = default;
  ItemConnectedListView(const impl::ItemIndexedListView<DynExtent>& view)
  : m_local_ids(view.constLocalIds()), m_shared_info(view.m_shared_info) { }

 protected:

  ItemConnectedListView(ItemSharedInfo* shared_info,ConstArrayView<Int32> local_ids)
  : m_local_ids(local_ids), m_shared_info(shared_info) { }

 public:

  // Temporaire pour rendre les sources compatibles
  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_shared_info,m_local_ids);
  }

  //! Accède au \a i-ème élément du vecteur
  Item operator[](Integer index) const { return Item(m_local_ids[index],m_shared_info); }

  //! Nombre d'éléments du vecteur
  Int32 size() const { return m_local_ids.size(); }

  // TODO: changer le type de l'iterateur
  const_iterator begin() const
  {
    return const_iterator(m_shared_info,m_local_ids.data());
  }
  const_iterator end() const
  {
    return const_iterator(m_shared_info,m_local_ids.data()+this->size());
  }

 public:

  // TODO Rendre privés
 
  //! Tableau des numéros locaux des entités
  Int32ConstArrayView localIds() const { return m_local_ids; }

 private:

  //! Vue sur le tableau des indices
  ItemIndexArrayView indexes() const { return m_local_ids; }

 public:

  inline ItemEnumerator enumerator() const;

 protected:
  
  ItemIndexArrayView m_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 private:

  void _init(ItemInternalArrayView items)
  {
    m_shared_info = (size()>0 && !items.empty()) ? ItemInternalCompatibility::_getSharedInfo(items[0]) : ItemSharedInfo::nullInstance();
  }
  void _init2(IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un tableau typé d'entités.
 */
template<typename ItemType,int Extent>
class ItemConnectedListViewT
: public ItemConnectedListView<Extent>
{
  friend class ItemVectorT<ItemType>;
  friend class ItemEnumeratorBaseT<ItemType>;
  friend class ItemEnumerator;
  friend class Item;
  friend class ItemWithNodes;

  using BaseClass = ItemConnectedListView<Extent>;
  using BaseClass::m_shared_info;
  using BaseClass::m_local_ids;

 public:

  using const_iterator = ItemVectorViewConstIteratorT<ItemType>;
  using difference_type = std::ptrdiff_t;
  using value_type = ItemType;

 private:

  ItemConnectedListViewT() = default;
  ItemConnectedListViewT(const ItemConnectedListView<Extent>& rhs)
  : BaseClass(rhs) {}
  ItemConnectedListViewT(const impl::ItemIndexedListView<DynExtent>& rhs)
  : BaseClass(rhs) {}

 protected:

  ItemConnectedListViewT(ItemSharedInfo* shared_info,ConstArrayView<Int32> local_ids)
  : BaseClass(shared_info,local_ids) { }

 public:

  operator ItemVectorViewT<ItemType> () const { return ItemVectorViewT<ItemType>(m_shared_info,m_local_ids); }

 public:

  ItemType operator[](Integer index) const
  {
    return ItemType(m_local_ids[index],m_shared_info);
  }

 public:
  
  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info,m_local_ids.localIds());
  }
  inline const_iterator begin() const
  {
    return const_iterator(m_shared_info,m_local_ids.data());
  }
  inline const_iterator end() const
  {
    return const_iterator(m_shared_info,m_local_ids.data()+this->size());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
