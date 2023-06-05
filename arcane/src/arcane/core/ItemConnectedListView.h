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

#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
#define ARCANE_LOCALID_ADD_OFFSET(a) (this->m_local_id_offset + (a))
#define ARCANE_ARGS_AND_OFFSET(a,b,c) a,b,c
#else
#define ARCANE_LOCALID_ADD_OFFSET(a) (a)
#define ARCANE_ARGS_AND_OFFSET(a,b,c) a,b
#endif

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
 * for( Item item : node.cell() )
 *    ;
 * \endcode
 */
class ItemConnectedListViewConstIterator
{
 protected:

  template<int Extent> friend class ItemConnectedListView;
  friend class ItemVectorViewConstIterator;

 protected:

#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
  ItemConnectedListViewConstIterator(ItemSharedInfo* shared_info,const Int32* local_id_ptr,
                                     Int32 local_id_offset)
  : m_shared_info(shared_info), m_local_id_ptr(local_id_ptr), m_local_id_offset(local_id_offset){}
#else
  ItemConnectedListViewConstIterator(ItemSharedInfo* shared_info,const Int32* local_id_ptr)
  : m_shared_info(shared_info), m_local_id_ptr(local_id_ptr){}
#endif

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

  Item operator*() const { return Item(*m_local_id_ptr,m_shared_info); }
  ThatClass& operator++() { ++m_local_id_ptr; return (*this); }
  ThatClass& operator--() { --m_local_id_ptr; return (*this); }
  void operator+=(difference_type v) { m_local_id_ptr += v; }
  void operator-=(difference_type v) { m_local_id_ptr -= v; }
  difference_type operator-(const ThatClass& b) const
  {
    return this->m_local_id_ptr - b.m_local_id_ptr;
  }
  friend ThatClass operator-(const ThatClass& a,difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr - v;
    return ThatClass(ARCANE_ARGS_AND_OFFSET(a.m_shared_info,ptr,a.m_local_id_offset));
  }
  friend ThatClass operator+(const ThatClass& a,difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(ARCANE_ARGS_AND_OFFSET(a.m_shared_info,ptr,a.m_local_id_offset));
  }
  friend bool operator<(const ThatClass& lhs,const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr <= rhs.m_local_id_ptr;
  }
  //! Compare les indices d'itération de deux instances
  friend bool operator==(const ThatClass& lhs,const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr == rhs.m_local_id_ptr;
  }
  friend bool operator!=(const ThatClass& lhs,const ThatClass& rhs)
  {
    return !(lhs==rhs);
  }

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  Item operator->() const { return _itemInternal(); }

 protected:

  ItemSharedInfo* m_shared_info;
  const Int32* m_local_id_ptr;
#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
  Int32 m_local_id_offset = 0;
#endif

 protected:

  inline ItemInternal* _itemInternal() const
  {
    return m_shared_info->m_items_internal[ ARCANE_LOCALID_ADD_OFFSET((*m_local_id_ptr)) ];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType>
class ItemConnectedListViewConstIteratorT
: public ItemConnectedListViewConstIterator
{
  friend class ItemConnectedListViewT<ItemType>;

 private:

#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
  ItemConnectedListViewConstIteratorT(ItemSharedInfo* shared_info,const Int32* ARCANE_RESTRICT local_id_ptr,
                                      Int32 local_id_offset)
  : ItemConnectedListViewConstIterator(shared_info,local_id_ptr,local_id_offset){}
#else
  ItemConnectedListViewConstIteratorT(ItemSharedInfo* shared_info,const Int32* ARCANE_RESTRICT local_id_ptr)
  : ItemConnectedListViewConstIterator(shared_info,local_id_ptr){}
#endif

 public:

  typedef ItemConnectedListViewConstIteratorT<ItemType> ThatClass;
  typedef ItemType value_type;

 public:

  //TODO A supprimer avec le C++20
  typedef const Item* pointer;
  //TODO A supprimer avec le C++20
  typedef const Item& reference;

 public:

  ItemType operator*() const { return ItemType(*m_local_id_ptr,m_shared_info); }
  ThatClass& operator++() { ++m_local_id_ptr; return (*this); }
  ThatClass& operator--() { --m_local_id_ptr; return (*this); }
  difference_type operator-(const ThatClass& b) const
  {
    return this->m_local_id_ptr - b.m_local_id_ptr;
  }
  friend ThatClass operator-(const ThatClass& a,difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr - v;
    return ThatClass(a.m_shared_info,ptr);
  }
  friend ThatClass operator+(const ThatClass& a,difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(a.m_shared_info,ptr);
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  ItemType operator->() const { return this->_itemInternal(); }
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
template<int Extent>
class ItemConnectedListView
{
  friend ItemVector;
  friend class ItemEnumeratorBase;
  friend class ItemVectorView;
  friend class ItemConnectedEnumeratorBase;
  template<typename ItemType> friend class ItemEnumeratorBaseT;

 public:

  using const_iterator = ItemConnectedListViewConstIterator;
  using difference_type = std::ptrdiff_t;
  using value_type = Item;
  using reference_type = Item&;
  using const_reference_type = const Item&;

 public:

  ItemConnectedListView() = default;

 protected:

  ItemConnectedListView(const impl::ItemIndexedListView<DynExtent>& view)
  : m_local_ids(view.constLocalIds()), m_shared_info(view.m_shared_info) { }
  ItemConnectedListView(ItemSharedInfo* shared_info,ConstArrayView<Int32> local_ids,Int32 local_id_offset)
  : m_local_ids(local_ids), m_shared_info(shared_info), m_local_id_offset(local_id_offset) { }

 public:

  //! Accède au \a i-ème élément du vecteur
  Item operator[](Integer index) const
  {
    return Item(ARCANE_LOCALID_ADD_OFFSET(m_local_ids[index]),m_shared_info);
  }

  //! Nombre d'éléments du vecteur
  Int32 size() const { return m_local_ids.size(); }

  // TODO: changer le type de l'iterateur
  const_iterator begin() const
  {
    return const_iterator(ARCANE_ARGS_AND_OFFSET(m_shared_info,m_local_ids.data(),m_local_id_offset));
  }

  // TODO: changer le type de l'iterateur
  const_iterator end() const
  {
    return const_iterator(ARCANE_ARGS_AND_OFFSET(m_shared_info,(m_local_ids.data()+this->size()),m_local_id_offset));
  }

  friend std::ostream& operator<<(std::ostream& o,const ItemConnectedListView<Extent>& a)
  {
    o << a.m_local_ids.localIds();
    return o;
  }

  ARCANE_DEPRECATED_REASON("Y2023: Use iterator to get values or use operator[]")
  Int32ConstArrayView localIds() const { return m_local_ids; }

#ifdef ARCANE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
 private:
#else
 public:
#endif

  // Temporaire pour rendre les sources compatibles
  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_shared_info,m_local_ids);
  }

  // TODO: rendre obsolète
  inline ItemEnumerator enumerator() const;

 private:

  //! Vue sur le tableau des indices
  ItemIndexArrayView indexes() const { return m_local_ids; }

  //! Vue sur le tableau des indices
  Int32ConstArrayView _localIds() const { return m_local_ids; }

 protected:
  
  ItemIndexArrayView m_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
  Int32 m_local_id_offset = 0;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste d'entités connectées à une autre.
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
  friend class Node;
  friend class Edge;
  friend class Face;
  friend class Cell;
  friend class Particle;
  friend class DoF;
  template <typename T> friend class ItemConnectedEnumeratorBaseT;

  using BaseClass = ItemConnectedListView<Extent>;
  using BaseClass::m_shared_info;
  using BaseClass::m_local_ids;
#ifdef ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW
  using BaseClass::m_local_id_offset;
#endif

 public:

  using const_iterator = ItemConnectedListViewConstIteratorT<ItemType>;
  using difference_type = std::ptrdiff_t;
  using value_type = ItemType;

 private:

  ItemConnectedListViewT() = default;
  ItemConnectedListViewT(const ItemConnectedListView<Extent>& rhs)
  : BaseClass(rhs) {}
  ItemConnectedListViewT(const impl::ItemIndexedListView<DynExtent>& rhs)
  : BaseClass(rhs) {}

 protected:

  ItemConnectedListViewT(ItemSharedInfo* shared_info,ConstArrayView<Int32> local_ids, Int32 local_id_offset)
  : BaseClass(shared_info,local_ids,local_id_offset) { }

 public:

  ItemType operator[](Integer index) const
  {
    return ItemType(ARCANE_LOCALID_ADD_OFFSET(m_local_ids[index]),m_shared_info);
  }

 public:
  
  inline const_iterator begin() const
  {
    return const_iterator(ARCANE_ARGS_AND_OFFSET(m_shared_info,m_local_ids.data(),m_local_id_offset));
  }
  inline const_iterator end() const
  {
    return const_iterator(ARCANE_ARGS_AND_OFFSET(m_shared_info,(m_local_ids.data()+this->size()),m_local_id_offset));
  }

#ifdef ARCANE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
 private:
#else
 public:
#endif

  // TODO: rendre obsolète
  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info,m_local_ids.localIds());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

#undef ARCANE_LOCALID_ADD_OFFSET
#undef ARCANE_THAT_CLASS_AND_OFFSET

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
