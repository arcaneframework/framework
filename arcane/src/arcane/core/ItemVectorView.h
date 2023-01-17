﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVectorView.h                                            (C) 2000-2023 */
/*                                                                           */
/* Vue sur un vecteur (tableau indirect) d'entités.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMVECTORVIEW_H
#define ARCANE_ITEMVECTORVIEW_H
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
 * \internal
 * \brief Itérateur pour la classe ItemVectorView.
 *
 * Cette classe est interne à Arcane. Elle s'utilise via le for-range:
 *
 * \code
 * ItemVectorView view;
 * for( Item item : view )
 *    ;
 * \endcode
 */
class ItemVectorViewConstIterator
{
 protected:
  friend class ItemVectorView;
  ItemVectorViewConstIterator(ItemSharedInfo* shared_info,const Int32* local_id_ptr)
  : m_shared_info(shared_info), m_local_id_ptr(local_id_ptr){}
 public:

  typedef ItemVectorViewConstIterator ThatClass;
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
    return ThatClass(a.m_shared_info,ptr);
  }
  friend ThatClass operator+(const ThatClass& a,difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(a.m_shared_info,ptr);
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

 protected:

  inline ItemInternal* _itemInternal() const
  {
    return m_shared_info->m_items_internal[ *m_local_id_ptr ];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType>
class ItemVectorViewConstIteratorT
: public ItemVectorViewConstIterator
{
  friend class ItemVectorViewT<ItemType>;

 private:

  ItemVectorViewConstIteratorT(ItemSharedInfo* shared_info,const Int32* ARCANE_RESTRICT local_id_ptr)
  : ItemVectorViewConstIterator(shared_info,local_id_ptr){}

 public:

  typedef ItemVectorViewConstIteratorT<ItemType> ThatClass;
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
 * \brief Vue sur un vecteur d'entités.
 *
 * \warning la vue n'est valide que tant que le tableau associé n'est
 * pas modifié et que la famille d'entité associée à ce tableau n'est
 * elle même pas modifiée.
 */
class ARCANE_CORE_EXPORT ItemVectorView
{
  friend class ItemVector;

 public:

  using const_iterator = ItemVectorViewConstIterator;
  using difference_type = std::ptrdiff_t;
  using value_type = Item;
  using reference_type = Item&;
  using const_reference_type = const Item&;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorView(const ItemInternalArrayView& aitems,const Int32ConstArrayView& local_ids)
  : m_local_ids(local_ids) { _init(aitems); }

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorView(ItemInternalArrayView aitems,ItemIndexArrayView indexes)
  : m_local_ids(indexes) { _init(aitems); }

 public:

  ItemVectorView() = default;
  ItemVectorView(const ItemInternalVectorView& view)
  : m_local_ids(view.localIds()), m_shared_info(view.m_shared_info) { }
  ItemVectorView(ItemInfoListView item_info_list_view,ConstArrayView<Int32> local_ids)
  : m_local_ids(local_ids), m_shared_info(item_info_list_view.m_item_shared_info) { }
  ItemVectorView(ItemInfoListView item_info_list_view,ItemIndexArrayView indexes)
  : m_local_ids(indexes), m_shared_info(item_info_list_view.m_item_shared_info) { }
  ItemVectorView(IItemFamily* family,ConstArrayView<Int32> local_ids);
  ItemVectorView(IItemFamily* family,ItemIndexArrayView indexes);
  ItemVectorView(const impl::ItemIndexedListView<DynExtent>& view)
  : m_local_ids(view.constLocalIds()), m_shared_info(view.m_shared_info) { }

 protected:

  ItemVectorView(ItemSharedInfo* shared_info,ConstArrayView<Int32> local_ids)
  : m_local_ids(local_ids), m_shared_info(shared_info) { }

  // Temporaire pour éviter un avertissement de compilation lorsqu'on utilise le
  // constructeur obsolète de ItemVectorViewT
  ItemVectorView(const ItemInternalArrayView& aitems,const Int32ConstArrayView& local_ids,bool)
  : m_local_ids(local_ids) { _init(aitems); }

  // Temporaire pour éviter un avertissement de compilation lorsqu'on utilise le
  // constructeur obsolète de ItemVectorViewT
  ItemVectorView(ItemInternalArrayView aitems,ItemIndexArrayView indexes,bool)
  : m_local_ids(indexes) { _init(aitems); }

 public:

  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_shared_info,m_local_ids);
  }

  //! Accède au \a i-ème élément du vecteur
  Item operator[](Integer index) const { return Item(m_local_ids[index],m_shared_info); }

  //! Nombre d'éléments du vecteur
  Int32 size() const { return m_local_ids.size(); }

  //! Tableau des entités
  ARCANE_DEPRECATED_REASON("Y2022: Do not use this method")
  ItemInternalArrayView items() const { return m_shared_info->m_items_internal; }

  //! Tableau des numéros locaux des entités
  Int32ConstArrayView localIds() const { return m_local_ids; }

  //! Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments
  ItemVectorView subView(Integer abegin,Integer asize) const
  {
    return ItemVectorView(m_shared_info,m_local_ids.subView(abegin,asize));
  }
  const_iterator begin() const
  {
    return const_iterator(m_shared_info,m_local_ids.data());
  }
  const_iterator end() const
  {
    return const_iterator(m_shared_info,m_local_ids.data()+this->size());
  }
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
template<typename ItemType>
class ItemVectorViewT
: public ItemVectorView
{
  friend class ItemVectorT<ItemType>;

 public:

  using const_iterator = ItemVectorViewConstIteratorT<ItemType>;
  using difference_type = std::ptrdiff_t;
  using value_type = ItemType;
  //TODO a supprimer avec le C++20
  using reference_type = ItemType&;
  //TODO a supprimer avec le C++20
  using const_reference_type = const ItemType&;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorViewT(const ItemInternalArrayView& aitems,const Int32ConstArrayView& local_ids)
  : ItemVectorView(aitems,local_ids,true) {}

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorViewT(ItemInternalArrayView aitems,ItemIndexArrayView indexes)
  : ItemVectorView(aitems,indexes,true) {}

 public:

  ItemVectorViewT() = default;
  ItemVectorViewT(const ItemVectorView& rhs)
  : ItemVectorView(rhs) {}
  inline ItemVectorViewT(const ItemVectorT<ItemType>& rhs);
  ItemVectorViewT(const ItemInternalVectorView& rhs)
  : ItemVectorView(rhs) {}
  ItemVectorViewT(const impl::ItemIndexedListView<DynExtent>& rhs)
  : ItemVectorView(rhs) {}
  ItemVectorViewT(ItemInfoListView item_info_list_view,ConstArrayView<Int32> local_ids)
  : ItemVectorView(item_info_list_view,local_ids) {}
  ItemVectorViewT(ItemInfoListView item_info_list_view,ItemIndexArrayView indexes)
  : ItemVectorView(item_info_list_view,indexes) {}
  ItemVectorViewT(IItemFamily* family,ConstArrayView<Int32> local_ids)
  : ItemVectorView(family,local_ids) {}
  ItemVectorViewT(IItemFamily* family,ItemIndexArrayView indexes)
  : ItemVectorView(family,indexes) {}

 protected:

  ItemVectorViewT(ItemSharedInfo* shared_info,ConstArrayView<Int32> local_ids)
  : ItemVectorView(shared_info,local_ids) { }

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
