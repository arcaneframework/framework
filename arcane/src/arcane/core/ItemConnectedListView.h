// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectedListView.h                                     (C) 2000-2025 */
/*                                                                           */
/* Vue sur une liste d'entités connectés à une autre entité.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMCONNECTEDLISTVIEW_H
#define ARCANE_CORE_ITEMCONNECTEDLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternalVectorView.h"
#include "arcane/core/ItemIndexArrayView.h"
#include "arcane/core/ItemInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Itérateur pour la classe ItemConnectedListView.
 *
 * Cette classe est interne à Arcane. Elle s'utilise via le for-range:
 *
 * \code
 * Node node;
 * for( Item item : node.cells() )
 *    ;
 * \endcode
 */
class ItemConnectedListViewConstIterator
{
 protected:

  template <int Extent> friend class ItemConnectedListView;
  friend class ItemVectorViewConstIterator;

 protected:

  ItemConnectedListViewConstIterator(ItemSharedInfo* shared_info, const Int32* local_id_ptr,
                                     Int32 local_id_offset)
  : m_shared_info(shared_info)
  , m_local_id_ptr(local_id_ptr)
  , m_local_id_offset(local_id_offset)
  {}

 public:

  typedef ItemConnectedListViewConstIterator ThatClass;
  typedef std::random_access_iterator_tag iterator_category;
  //! Type indexant le tableau
  typedef Item value_type;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef std::ptrdiff_t difference_type;

 public:

  //TODO A supprimer avec le C++20
  typedef const Item* pointer;
  //TODO A supprimer avec le C++20
  typedef const Item& reference;

 public:

  Item operator*() const
  {
    return Item(*m_local_id_ptr, m_shared_info);
  }
  ThatClass& operator++()
  {
    ++m_local_id_ptr;
    return (*this);
  }
  ThatClass& operator--()
  {
    --m_local_id_ptr;
    return (*this);
  }
  void operator+=(difference_type v)
  {
    m_local_id_ptr += v;
  }
  void operator-=(difference_type v)
  {
    m_local_id_ptr -= v;
  }
  difference_type operator-(const ThatClass& b) const
  {
    return this->m_local_id_ptr - b.m_local_id_ptr;
  }
  friend ThatClass operator-(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr - v;
    return ThatClass(a.m_shared_info, ptr, a.m_local_id_offset);
  }
  friend ThatClass operator+(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(a.m_shared_info, ptr, a.m_local_id_offset);
  }
  friend bool operator<(const ThatClass& lhs, const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr <= rhs.m_local_id_ptr;
  }
  //! Compare les indices d'itération de deux instances
  friend bool operator==(const ThatClass& lhs, const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr == rhs.m_local_id_ptr;
  }
  friend bool operator!=(const ThatClass& lhs, const ThatClass& rhs)
  {
    return !(lhs == rhs);
  }

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  Item operator->() const
  {
    return _itemInternal();
  }

 protected:

  ItemSharedInfo* m_shared_info;
  const Int32* m_local_id_ptr;
  Int32 m_local_id_offset = 0;

 protected:

  inline ItemInternal* _itemInternal() const
  {
    return m_shared_info->m_items_internal[m_local_id_offset + (*m_local_id_ptr)];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType>
class ItemConnectedListViewConstIteratorT
: public ItemConnectedListViewConstIterator
{
  friend class ItemConnectedListViewT<ItemType>;

 private:

  ItemConnectedListViewConstIteratorT(ItemSharedInfo* shared_info, const Int32* ARCANE_RESTRICT local_id_ptr,
                                      Int32 local_id_offset)
  : ItemConnectedListViewConstIterator(shared_info, local_id_ptr, local_id_offset)
  {}

 public:

  typedef ItemConnectedListViewConstIteratorT<ItemType> ThatClass;
  typedef ItemType value_type;

 public:

  //TODO A supprimer avec le C++20
  typedef const Item* pointer;
  //TODO A supprimer avec le C++20
  typedef const Item& reference;

 public:

  ItemType operator*() const
  {
    return ItemType(*m_local_id_ptr, m_shared_info);
  }
  ThatClass& operator++()
  {
    ++m_local_id_ptr;
    return (*this);
  }
  ThatClass& operator--()
  {
    --m_local_id_ptr;
    return (*this);
  }
  difference_type operator-(const ThatClass& b) const
  {
    return this->m_local_id_ptr - b.m_local_id_ptr;
  }
  friend ThatClass operator-(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr - v;
    return ThatClass(a.m_shared_info, ptr);
  }
  friend ThatClass operator+(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(a.m_shared_info, ptr);
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  ItemType operator->() const
  {
    return this->_itemInternal();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste d'entités connectées à une autre entité.
 *
 * \warning la vue n'est valide que tant que le tableau associé n'est
 * pas modifié et que la famille d'entité associée à ce tableau n'est
 * elle même pas modifiée.
 */
template <int Extent>
class ItemConnectedListView
{
  friend ItemVector;
  friend class ItemEnumeratorBase;
  friend class ItemVectorView;
  friend class ItemConnectedEnumeratorBase;
  template <typename ItemType> friend class ItemEnumeratorBaseT;

 public:

  using const_iterator = ItemConnectedListViewConstIterator;
  using difference_type = std::ptrdiff_t;
  using value_type = Item;
  using reference_type = Item&;
  using const_reference_type = const Item&;
  // TODO: Créér le type 'Sentinel' lorsqu'on sera en C++20
  using SentinelType = const_iterator;

 public:

  ItemConnectedListView() = default;

 protected:

  ItemConnectedListView(const impl::ItemIndexedListView<DynExtent>& view)
  : m_index_view(view.m_local_ids,view.m_local_id_offset,0)
  , m_shared_info(view.m_shared_info)
  {}
  ItemConnectedListView(ItemSharedInfo* shared_info, ConstArrayView<Int32> local_ids, Int32 local_id_offset)
  : m_index_view(local_ids,local_id_offset,0)
  , m_shared_info(shared_info)
  {}

 public:

  //! \a index-ème entité connectée
  Item operator[](Integer index) const
  {
    return Item(m_index_view[index], m_shared_info);
  }

  //! Nombre d'éléments du vecteur
  Int32 size() const { return m_index_view.size(); }

  //! Itérateur sur la première entité connectée
  const_iterator begin() const
  {
    return const_iterator(m_shared_info, m_index_view._data(), _localIdOffset());
  }

  //! Itérateur sur après la dernière entité connectée
  SentinelType end() const
  {
    return endIterator();
  }

  //! Itérateur sur après la dernière entité connectée
  const_iterator endIterator() const
  {
    return const_iterator(m_shared_info, (m_index_view._data() + this->size()), _localIdOffset());
  }

  friend std::ostream& operator<<(std::ostream& o, const ItemConnectedListView<Extent>& a)
  {
    o << a.m_index_view;
    return o;
  }

  ARCANE_DEPRECATED_REASON("Y2023: Use iterator to get values or use operator[]")
  Int32ConstArrayView localIds() const { return m_index_view._localIds(); }

#ifdef ARCANE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
 private:

#else
 public:
#endif

  // Temporaire pour rendre les sources compatibles
  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_shared_info, m_index_view._localIds(), _localIdOffset());
  }

  // TODO: rendre obsolète
  inline ItemEnumerator enumerator() const;

 private:

  //! Vue sur le tableau des indices
  ItemIndexArrayView indexes() const
  {
    return m_index_view;
  }

  //! Vue sur le tableau des indices
  Int32ConstArrayView _localIds() const
  {
    return m_index_view._localIds();
  }

 protected:

  ItemIndexArrayView m_index_view;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 protected:

  const Int32* _localIdsData() const { return m_index_view._data(); }
  Int32 _localIdOffset() const { return m_index_view._localIdOffset(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste d'entités connectées à une autre.
 */
template <typename ItemType, int Extent>
class ItemConnectedListViewT
: public ItemConnectedListView<Extent>
{
  friend class ItemVectorT<ItemType>;
  friend class ItemEnumeratorBaseT<ItemType>;
  friend class ItemEnumerator;
  friend class Item;
  friend class ItemWithNodes;
  friend class Node;
  friend class Edge;
  friend class Face;
  friend class Cell;
  friend class Particle;
  friend class DoF;
  template <typename T> friend class ItemConnectedEnumeratorBaseT;

  using BaseClass = ItemConnectedListView<Extent>;
  using BaseClass::m_index_view;
  using BaseClass::m_shared_info;

 public:

  using const_iterator = ItemConnectedListViewConstIteratorT<ItemType>;
  using difference_type = std::ptrdiff_t;
  using value_type = ItemType;
  // TODO: Créér le type 'Sentinel' lorsqu'on sera en C++20
  using SentinelType = const_iterator;

 private:

  ItemConnectedListViewT() = default;
  ItemConnectedListViewT(const ItemConnectedListView<Extent>& rhs)
  : BaseClass(rhs)
  {}
  ItemConnectedListViewT(const impl::ItemIndexedListView<DynExtent>& rhs)
  : BaseClass(rhs)
  {}

 protected:

  ItemConnectedListViewT(ItemSharedInfo* shared_info, ConstArrayView<Int32> local_ids, Int32 local_id_offset)
  : BaseClass(shared_info, local_ids, local_id_offset)
  {}

 public:

  //! \a index-ème entité connectée
  ItemType operator[](Integer index) const
  {
    return ItemType(m_index_view[index], m_shared_info);
  }

 public:

  //! Itérateur sur la première entité connectée
  inline const_iterator begin() const
  {
    return const_iterator(m_shared_info, this->_localIdsData(), this->_localIdOffset());
  }
  //! Itérateur sur après la dernière entité connectée
  inline SentinelType end() const
  {
    return endIterator();
  }
  //! Itérateur sur après la dernière entité connectée
  inline const_iterator endIterator() const
  {
    return const_iterator(m_shared_info, (this->_localIdsData() + this->size()), this->_localIdOffset());
  }

#ifdef ARCANE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
 private:

#else
 public:
#endif

  // TODO: rendre obsolète
  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info, m_index_view.m_view);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
