// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVectorView.h                                            (C) 2000-2022 */
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

class ItemEnumerator;
class ItemVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemVectorViewConstIterator
{
 protected:
  friend class ItemVectorView;
  typedef ItemInternal* ItemInternalPtr;
  ItemVectorViewConstIterator(ItemSharedInfo* shared_info,
                              const Int32* ARCANE_RESTRICT local_ids,
                              Integer index)
  : m_shared_info(shared_info), m_local_ids(local_ids), m_index(index){}
 public:
  typedef ItemVectorViewConstIterator ThatClass;
 public:
  typedef std::random_access_iterator_tag iterator_category;
  //! Type indexant le tableau
  typedef const Item* pointer;
  //! Type indexant le tableau
  typedef const Item& reference;
  //! Type indexant le tableau
  typedef Item value_type;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef Integer difference_type;

 public:

  Item operator*() const { return Item(m_local_ids[m_index],m_shared_info); }
  ThatClass& operator++() { ++m_index; return (*this); }
  ThatClass& operator--() { --m_index; return (*this); }
  void operator+=(difference_type v) { m_index += v; }
  void operator-=(difference_type v) { m_index -= v; }
  Integer operator-(const ThatClass& b) const
  {
    return this->m_index - b.m_index;
  }
  friend ThatClass operator-(const ThatClass& a,difference_type v)
  {
    Integer index = a.m_index - v;
    return ThatClass(a.m_shared_info,a.m_local_ids,index);
  }
  friend ThatClass operator+(const ThatClass& a,difference_type v)
  {
    Integer index = a.m_index + v;
    return ThatClass(a.m_shared_info,a.m_local_ids,index);
  }
  friend bool operator<(const ThatClass& lhs,const ThatClass& rhs)
  {
    return lhs.m_index<=rhs.m_index;
  }
  friend bool operator==(const ThatClass& lhs,const ThatClass& rhs)
  {
    return lhs.m_shared_info==rhs.m_shared_info && lhs.m_local_ids==rhs.m_local_ids && lhs.m_index==rhs.m_index;
  }
  friend bool operator!=(const ThatClass& lhs,const ThatClass& rhs)
  {
    return !(lhs==rhs);
  }

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  Item operator->() const { return _itemInternal(); }

 protected:

  ItemSharedInfo* m_shared_info;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;

 protected:

  inline ItemInternal* _itemInternal() const
  {
    return m_shared_info->m_items_internal[ m_local_ids[m_index] ];
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

  // Autorisé seulement par ItemVectorViewT.
  ItemVectorViewConstIteratorT(ItemSharedInfo* shared_info,
                               const Int32* ARCANE_RESTRICT local_ids,
                               Integer index)
  : ItemVectorViewConstIterator(shared_info,local_ids,index){}

 public:

  typedef ItemVectorViewConstIteratorT<ItemType> ThatClass;

 public:

  ItemType operator*() const { return ItemType(m_local_ids[m_index],m_shared_info); }
  ThatClass& operator++() { ++m_index; return (*this); }
  ThatClass& operator--() { --m_index; return (*this); }
  Integer operator-(const ThatClass& b) const
  {
    return this->m_index - b.m_index;
  }
  ThatClass operator-(difference_type v) const
  {
    Integer index = m_index - v;
    return ThatClass(m_shared_info,m_local_ids,index);
  }
  ThatClass operator+(difference_type v) const
  {
    Integer index = m_index + v;
    return ThatClass(m_shared_info,m_local_ids,index);
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
 public:

  using const_iterator = ItemVectorViewConstIterator;

 public:

  // TODO: a supprimer dès qu'on n'aura plus besoin de ItemInternal
  ItemVectorView(const ItemInternalArrayView& aitems,const Int32ConstArrayView& local_ids)
  : m_local_ids(local_ids) { _init(aitems); }
  // TODO: a supprimer dès qu'on n'aura plus besoin de ItemInternal
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

 private:

  ItemVectorView(ItemSharedInfo* shared_info,const Int32ConstArrayView& local_ids)
  : m_local_ids(local_ids), m_shared_info(shared_info) { }

 public:

  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_shared_info,m_local_ids);
  }

  //! Accède au \a i-ème élément du vecteur
  inline Item operator[](Integer index) const
  {
    return Item(m_local_ids[index],m_shared_info);
  }

  //! Nombre d'éléments du vecteur
  inline Integer size() const { return m_local_ids.size(); }

  //! Tableau des entités
  inline ItemInternalArrayView items() const { return m_shared_info->m_items_internal; }

  //! Tableau des numéros locaux des entités
  inline Int32ConstArrayView localIds() const { return m_local_ids; }

  //! Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments
  inline ItemVectorView subView(Integer abegin,Integer asize)
  {
    return ItemVectorView(m_shared_info,m_local_ids.subView(abegin,asize));
  }
  inline const_iterator begin() const
  {
    return const_iterator(m_shared_info,m_local_ids.data(),0);
  }
  inline const_iterator end() const
  {
    return const_iterator(m_shared_info,m_local_ids.data(),this->size());
  }
  //! Vue sur le tableau des indices
  inline ItemIndexArrayView indexes() const { return m_local_ids; }

 public:

  inline ItemEnumerator enumerator() const;

 protected:
  
  ItemIndexArrayView m_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 private:

  void _init(ItemInternalArrayView items)
  {
    m_shared_info = (size()>0 && !items.empty()) ? items[0]->sharedInfo() : ItemSharedInfo::nullInstance();
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
 public:

  using const_iterator = ItemVectorViewConstIteratorT<ItemType>;

 public:

  // TODO: a supprimer dès qu'on n'aura plus besoin de ItemInternal
  ItemVectorViewT(const ItemInternalArrayView& aitems,const Int32ConstArrayView& local_ids)
  : ItemVectorView(aitems,local_ids) {}
  // TODO: a supprimer dès qu'on n'aura plus besoin de ItemInternal
  ItemVectorViewT(ItemInternalArrayView aitems,ItemIndexArrayView indexes)
  : ItemVectorView(aitems,indexes) {}

 public:

  ItemVectorViewT() = default;
  ItemVectorViewT(const ItemVectorView& rhs)
  : ItemVectorView(rhs) {}
  inline ItemVectorViewT(const ItemVectorT<ItemType>& rhs);
  ItemVectorViewT(const ItemInternalVectorView& rhs)
  : ItemVectorView(rhs) {}
  ItemVectorViewT(ItemInfoListView item_info_list_view,ConstArrayView<Int32> local_ids)
  : ItemVectorView(item_info_list_view,local_ids) {}
  ItemVectorViewT(ItemInfoListView item_info_list_view,ItemIndexArrayView indexes)
  : ItemVectorView(item_info_list_view,indexes) {}
  ItemVectorViewT(IItemFamily* family,ConstArrayView<Int32> local_ids)
  : ItemVectorView(family,local_ids) {}
  ItemVectorViewT(IItemFamily* family,ItemIndexArrayView indexes)
  : ItemVectorView(family,indexes) {}

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
    return const_iterator(m_shared_info,m_local_ids.data(),0);
  }
  inline const_iterator end() const
  {
    return const_iterator(m_shared_info,m_local_ids.data(),this->size());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
