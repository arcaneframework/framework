// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVectorView.h                                            (C) 2000-2024 */
/*                                                                           */
/* View on a vector (indirect array) of entities.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMVECTORVIEW_H
#define ARCANE_CORE_ITEMVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternalVectorView.h"
#include "arcane/core/ItemIndexArrayView.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/ItemConnectedListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// For the C# wrapper
class ItemVectorViewPOD;
namespace mesh
{
  class IndexedItemConnectivityAccessor;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Iterator for the ItemVectorView class.
 *
 * This class is internal to Arcane. It is used via the for-range:
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
  template <int Extent> friend class ItemConnectedListView;

 protected:

  ItemVectorViewConstIterator(ItemSharedInfo* shared_info, const Int32* local_id_ptr, Int32 local_id_offset)
  : m_shared_info(shared_info)
  , m_local_id_ptr(local_id_ptr)
  , m_local_id_offset(local_id_offset)
  {}
  ItemVectorViewConstIterator(ItemSharedInfo* shared_info, const Int32* local_id_ptr)
  : m_shared_info(shared_info)
  , m_local_id_ptr(local_id_ptr)
  {}

 public:

  // Temporary (01/2023) for conversion with the new ItemConnectedListView type
  ItemVectorViewConstIterator(const ItemConnectedListViewConstIterator& v)
  : m_shared_info(v.m_shared_info)
  , m_local_id_ptr(v.m_local_id_ptr)
  , m_local_id_offset(v.m_local_id_offset)
  {}

 public:

  typedef ItemVectorViewConstIterator ThatClass;
  typedef std::random_access_iterator_tag iterator_category;
  //! Type indexing the array
  typedef Item value_type;
  //! Type indexing the array
  typedef Integer size_type;
  //! Type of a distance between iterator elements of the array
  typedef std::ptrdiff_t difference_type;

 public:

  //TODO To be removed with C++20
  typedef const Item* pointer;
  //TODO To be removed with C++20
  typedef const Item& reference;

 public:

  Item operator*() const { return Item(m_local_id_offset + (*m_local_id_ptr), m_shared_info); }

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
  void operator+=(difference_type v) { m_local_id_ptr += v; }
  void operator-=(difference_type v) { m_local_id_ptr -= v; }
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
  //! Compare the iteration indices of two instances
  friend bool operator==(const ThatClass& lhs, const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr == rhs.m_local_id_ptr;
  }
  friend bool operator!=(const ThatClass& lhs, const ThatClass& rhs)
  {
    return !(lhs == rhs);
  }

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  Item operator->() const { return _itemInternal(); }

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
class ItemVectorViewConstIteratorT
: public ItemVectorViewConstIterator
{
  friend class ItemVectorViewT<ItemType>;
  friend class ItemConnectedListViewT<ItemType>;

 private:

  ItemVectorViewConstIteratorT(ItemSharedInfo* shared_info, const Int32* ARCANE_RESTRICT local_id_ptr,
                               Int32 local_id_offset)
  : ItemVectorViewConstIterator(shared_info, local_id_ptr, local_id_offset)
  {}
  ItemVectorViewConstIteratorT(ItemSharedInfo* shared_info, const Int32* ARCANE_RESTRICT local_id_ptr)
  : ItemVectorViewConstIterator(shared_info, local_id_ptr)
  {}

 public:

  // Temporary (01/2023) for conversion with the new ItemConnectedListView type
  ItemVectorViewConstIteratorT(const ItemConnectedListViewConstIteratorT<ItemType>& v)
  : ItemVectorViewConstIterator(v)
  {}

 public:

  typedef ItemVectorViewConstIteratorT<ItemType> ThatClass;
  typedef ItemType value_type;

 public:

  //TODO To be removed with C++20
  typedef const Item* pointer;
  //TODO To be removed with C++20
  typedef const Item& reference;

 public:

  ItemType operator*() const { return ItemType(m_local_id_offset + (*m_local_id_ptr), m_shared_info); }

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
    return ThatClass(a.m_shared_info, ptr, a.m_local_id_offset);
  }
  friend ThatClass operator+(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(a.m_shared_info, ptr, a.m_local_id_offset);
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method returns a temporary. Use 'operator*' instead")
  ItemType operator->() const { return this->_itemInternal(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View on a vector of entities.
 *
 * \warning the view is only valid as long as the associated array is not
 * modified and the entity family associated with this array is not
 * modified itself.
 */
class ARCANE_CORE_EXPORT ItemVectorView
{
  // NOTE: This class is mapped to C# and if its structure changes,
  // the corresponding C# version must be updated.

  friend ItemVector;
  friend ItemEnumeratorBase;
  friend mesh::IndexedItemConnectivityAccessor;

 public:

  using const_iterator = ItemVectorViewConstIterator;
  using difference_type = std::ptrdiff_t;
  using value_type = Item;
  using reference_type = Item&;
  using const_reference_type = const Item&;
  // TODO: Create the 'Sentinel' type when we are in C++20
  using SentinelType = const_iterator;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorView(const ItemInternalArrayView& aitems, const Int32ConstArrayView& local_ids)
  : m_index_view(local_ids)
  {
    _init(aitems);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorView(ItemInternalArrayView aitems, ItemIndexArrayView indexes)
  : m_index_view(indexes)
  {
    _init(aitems);
  }

 public:

  ItemVectorView() = default;
  // TODO Deprecate (3.11+)
  ItemVectorView(const ItemInternalVectorView& view)
  : m_index_view(view.localIds(), view.m_local_id_offset, 0)
  , m_shared_info(view.m_shared_info)
  {}
  // TODO To be removed
  ItemVectorView(ItemInfoListView item_info_list_view, ConstArrayView<Int32> local_ids)
  : m_index_view(local_ids, 0, 0)
  , m_shared_info(item_info_list_view.m_item_shared_info)
  {}
  ItemVectorView(ItemInfoListView item_info_list_view, ItemIndexArrayView indexes)
  : m_index_view(indexes)
  , m_shared_info(item_info_list_view.m_item_shared_info)
  {}
  ItemVectorView(IItemFamily* family, ConstArrayView<Int32> local_ids);
  ItemVectorView(IItemFamily* family, ItemIndexArrayView indexes);
  ItemVectorView(const impl::ItemIndexedListView<DynExtent>& view)
  : m_index_view(view.constLocalIds(), view.m_local_id_offset, 0)
  , m_shared_info(view.m_shared_info)
  {}

  // Temporary (01/2023) for conversion with the new ItemConnectedListView type
  ItemVectorView(const ItemConnectedListView<DynExtent>& v)
  : m_index_view(v.m_index_view)
  , m_shared_info(v.m_shared_info)
  {}

 protected:

  ItemVectorView(ItemSharedInfo* shared_info, const impl::ItemLocalIdListContainerView& local_ids)
  : m_index_view(local_ids)
  , m_shared_info(shared_info)
  {}

  ItemVectorView(ItemSharedInfo* shared_info, ConstArrayView<Int32> local_ids, Int32 local_id_offset)
  : m_index_view(local_ids, local_id_offset, 0)
  , m_shared_info(shared_info)
  {}

  // Temporary to avoid a compilation warning when using the
  // deprecated constructor of ItemVectorViewT
  ItemVectorView(const ItemInternalArrayView& aitems, const Int32ConstArrayView& local_ids, bool)
  : m_index_view(local_ids)
  {
    _init(aitems);
  }

  // Temporary to avoid a compilation warning when using the
  // deprecated constructor of ItemVectorViewT
  ItemVectorView(ItemInternalArrayView aitems, ItemIndexArrayView indexes, bool)
  : m_index_view(indexes)
  {
    _init(aitems);
  }

 public:

  // TODO to be removed
  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_shared_info, m_index_view._localIds(), _localIdOffset());
  }

  //! Access the i-th element of the vector
  Item operator[](Integer index) const { return Item(m_index_view[index], m_shared_info); }

  //! Number of elements in the vector
  Int32 size() const { return m_index_view.size(); }

  //! Array of entities
  ARCANE_DEPRECATED_REASON("Y2022: Do not use this method")
  ItemInternalArrayView items() const { return m_shared_info->m_items_internal; }

  // TODO Deprecate (3.11+)
  /*!
   * \brief Array of local IDs of entities.
   *
   * \deprecated Do not retrieve the list of entities directly.
   * It is preferable to use iterators or the method
   * fillLocalIds() if you want to retrieve the list of localIds().
   */
  Int32ConstArrayView localIds() const { return m_index_view._localIds(); }

  //! Adds the list of localIds() of the vector to ids.
  void fillLocalIds(Array<Int32>& ids) const;

  //! Sub-view starting from element abegin and containing asize elements
  ItemVectorView subView(Integer abegin, Integer asize) const
  {
    return ItemVectorView(m_shared_info, m_index_view.subView(abegin, asize)._localIds(), _localIdOffset());
  }
  const_iterator begin() const
  {
    return const_iterator(m_shared_info, m_index_view._data(), _localIdOffset());
  }
  SentinelType end() const
  {
    return endIterator();
  }
  const_iterator endIterator() const
  {
    return const_iterator(m_shared_info, (m_index_view._data() + this->size()), _localIdOffset());
  }
  //! View on the array of indices
  ItemIndexArrayView indexes() const { return m_index_view; }

 public:

  inline ItemEnumerator enumerator() const;

 protected:

  ItemIndexArrayView m_index_view;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 protected:

  const Int32* _localIdsData() const { return m_index_view._data(); }
  Int32 _localIdOffset() const { return m_index_view._localIdOffset(); }

 private:

  void _init(ItemInternalArrayView items)
  {
    m_shared_info = (size() > 0 && !items.empty()) ? ItemInternalCompatibility::_getSharedInfo(items[0]) : ItemSharedInfo::nullInstance();
  }
  void _init2(IItemFamily* family);

 public:

  // For SWIG to position the values in the POD instance
  void _internalSwigSet(ItemVectorViewPOD* vpod);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View on a typed array of entities.
 */
template <typename ItemType>
class ItemVectorViewT
: public ItemVectorView
{
  friend class ItemVectorT<ItemType>;
  friend class ItemConnectedListViewT<ItemType>;

 public:

  using const_iterator = ItemVectorViewConstIteratorT<ItemType>;
  using difference_type = std::ptrdiff_t;
  using value_type = ItemType;
  //TODO to be removed with C++20
  using reference_type = ItemType&;
  //TODO to be removed with C++20
  using const_reference_type = const ItemType&;
  // TODO: Create the 'Sentinel' type when we are in C++20
  using SentinelType = const_iterator;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorViewT(const ItemInternalArrayView& aitems, const Int32ConstArrayView& local_ids)
  : ItemVectorView(aitems, local_ids, true)
  {}

  ARCANE_DEPRECATED_REASON("Y2022: Use constructor with ItemInfoListView instead")
  ItemVectorViewT(ItemInternalArrayView aitems, ItemIndexArrayView indexes)
  : ItemVectorView(aitems, indexes, true)
  {}

 public:

  ItemVectorViewT() = default;
  ItemVectorViewT(const ItemVectorView& rhs)
  : ItemVectorView(rhs)
  {}
  inline ItemVectorViewT(const ItemVectorT<ItemType>& rhs);
  ItemVectorViewT(const ItemInternalVectorView& rhs)
  : ItemVectorView(rhs)
  {}
  ItemVectorViewT(const impl::ItemIndexedListView<DynExtent>& rhs)
  : ItemVectorView(rhs)
  {}
  ItemVectorViewT(ItemInfoListView item_info_list_view, ConstArrayView<Int32> local_ids)
  : ItemVectorView(item_info_list_view, local_ids)
  {}
  ItemVectorViewT(ItemInfoListView item_info_list_view, ItemIndexArrayView indexes)
  : ItemVectorView(item_info_list_view, indexes)
  {}
  ItemVectorViewT(IItemFamily* family, ConstArrayView<Int32> local_ids)
  : ItemVectorView(family, local_ids)
  {}
  ItemVectorViewT(IItemFamily* family, ItemIndexArrayView indexes)
  : ItemVectorView(family, indexes)
  {}

  // Temporary (01/2023) for conversion with the new ItemConnectedListView type
  ItemVectorViewT(const ItemConnectedListViewT<ItemType>& v)
  : ItemVectorView(v)
  {}

 protected:

  ItemVectorViewT(ItemSharedInfo* shared_info, ConstArrayView<Int32> local_ids, Int32 local_id_offset)
  : ItemVectorView(shared_info, local_ids, local_id_offset)
  {}

 public:

  ItemType operator[](Integer index) const
  {
    return ItemType(m_index_view[index], m_shared_info);
  }

 public:

  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info, m_index_view.m_view);
  }
  inline const_iterator begin() const
  {
    return const_iterator(m_shared_info, _localIdsData(), _localIdOffset());
  }
  inline SentinelType end() const
  {
    return const_iterator(m_shared_info, _localIdsData() + this->size(), _localIdOffset());
  }
  inline const_iterator endIterator() const
  {
    return const_iterator(m_shared_info, _localIdsData() + this->size(), _localIdOffset());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
