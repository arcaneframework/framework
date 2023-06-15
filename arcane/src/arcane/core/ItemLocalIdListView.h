// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalIdListView.h                                       (C) 2000-2023 */
/*                                                                           */
/* Vue sur une liste de ItemLocalId.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMLOCALIDLISTVIEW_H
#define ARCANE_CORE_ITEMLOCALIDLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemLocalId.h"
#include "arcane/core/ItemLocalIdListContainerView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
class MeshUnitTest;
}

namespace Arcane
{
namespace mesh
{
  class IndexedItemConnectivityAccessor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des itérateurs pour ItemLocalIdViewT.
 */
class ItemLocalIdListViewConstIterator
{
 protected:

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListViewConstIterator(const Int32* local_id_ptr, Int32 local_id_offset)
  : m_local_id_ptr(local_id_ptr)
  , m_local_id_offset(local_id_offset)
  {}

 public:

  typedef ItemLocalIdListViewConstIterator ThatClass;
  typedef std::random_access_iterator_tag iterator_category;
  //! Type indexant le tableau
  typedef Int32 value_type;
  //! Type de la taille
  typedef Int32 size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef std::ptrdiff_t difference_type;

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 operator*() const { return m_local_id_offset + (*m_local_id_ptr); }

  constexpr ARCCORE_HOST_DEVICE ThatClass& operator++()
  {
    ++m_local_id_ptr;
    return (*this);
  }
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator--()
  {
    --m_local_id_ptr;
    return (*this);
  }
  constexpr ARCCORE_HOST_DEVICE void operator+=(difference_type v) { m_local_id_ptr += v; }
  constexpr ARCCORE_HOST_DEVICE void operator-=(difference_type v) { m_local_id_ptr -= v; }
  constexpr ARCCORE_HOST_DEVICE difference_type operator-(const ThatClass& b) const
  {
    return this->m_local_id_ptr - b.m_local_id_ptr;
  }
  constexpr ARCCORE_HOST_DEVICE friend ThatClass operator-(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr - v;
    return ThatClass(ptr, a.m_local_id_offset);
  }
  constexpr ARCCORE_HOST_DEVICE friend ThatClass operator+(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(ptr, a.m_local_id_offset);
  }
  constexpr ARCCORE_HOST_DEVICE friend bool operator<(const ThatClass& lhs, const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr <= rhs.m_local_id_ptr;
  }
  //! Compare les indices d'itération de deux instances
  constexpr ARCCORE_HOST_DEVICE friend bool operator==(const ThatClass& lhs, const ThatClass& rhs)
  {
    return lhs.m_local_id_ptr == rhs.m_local_id_ptr;
  }
  constexpr ARCCORE_HOST_DEVICE friend bool operator!=(const ThatClass& lhs, const ThatClass& rhs)
  {
    return !(lhs == rhs);
  }

 protected:

  const Int32* m_local_id_ptr;
  Int32 m_local_id_offset = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Itérateur pour ItemLocalIdViewT.
 */
template <typename ItemType>
class ItemLocalIdListViewConstIteratorT
: public ItemLocalIdListViewConstIterator
{
  friend class ItemLocalIdViewT<ItemType>;

 private:

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListViewConstIteratorT(const Int32* local_id_ptr, Int32 local_id_offset)
  : ItemLocalIdListViewConstIterator(local_id_ptr, local_id_offset)
  {}

 public:

  using LocalIdType = typename ItemLocalIdTraitsT<ItemType>::LocalIdType;
  using ThatClass = ItemLocalIdListViewConstIteratorT<ItemType>;
  using value_type = LocalIdType;

 public:

  constexpr ARCCORE_HOST_DEVICE LocalIdType operator*() const { return LocalIdType(m_local_id_offset + (*m_local_id_ptr)); }

  constexpr ARCCORE_HOST_DEVICE ThatClass& operator++()
  {
    ++m_local_id_ptr;
    return (*this);
  }
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator--()
  {
    --m_local_id_ptr;
    return (*this);
  }
  constexpr ARCCORE_HOST_DEVICE difference_type operator-(const ThatClass& b) const
  {
    return this->m_local_id_ptr - b.m_local_id_ptr;
  }
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr - v;
    return ThatClass(a.m_shared_info, ptr, a.m_local_id_offset);
  }
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, difference_type v)
  {
    const Int32* ptr = a.m_local_id_ptr + v;
    return ThatClass(a.m_shared_info, ptr, a.m_local_id_offset);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste de ItemLocalId.
 */
class ARCANE_CORE_EXPORT ItemLocalIdListView
: private impl::ItemLocalIdListContainerView
{
  template <typename ItemType> friend class ItemLocalIdViewT;
  friend class ItemVectorView;
  using impl::ItemLocalIdListContainerView::m_size;
  using impl::ItemLocalIdListContainerView::localId;

 public:

  using ThatClass = ItemLocalIdListView;
  using BaseClass = impl::ItemLocalIdListContainerView;

 private:

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListView(const Int32* ids, Int32 s, Int32 local_id_offset)
  : BaseClass(ids,s,local_id_offset)
  {}

 public:

  ARCCORE_HOST_DEVICE ItemLocalId operator[](Int32 index) const
  {
    return ItemLocalId(localId(index));
  }
  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_size; }

 public:

  friend ARCANE_CORE_EXPORT bool operator==(const ThatClass& lhs, const ThatClass& rhs);
  friend inline bool operator!=(const ThatClass& lhs, const ThatClass& rhs)
  {
    return !operator==(lhs, rhs);
  }
  friend ARCANE_CORE_EXPORT std::ostream& operator<<(std::ostream& o, const ThatClass& lhs);

 private:

  ConstArrayView<Int32> _idsWithoutOffset() const { return BaseClass::_idsWithoutOffset(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue typée sur une liste d'entités d'une connectivité.
 */
template <typename ItemType>
class ItemLocalIdViewT
: public ItemLocalIdListView
{
  friend class ItemConnectivityContainerView;
  friend mesh::IndexedItemConnectivityAccessor;
  friend ArcaneTest::MeshUnitTest;
  friend class Item;
  friend class ItemInternalConnectivityList;

 public:

  using LocalIdType = typename ItemLocalIdTraitsT<ItemType>::LocalIdType;
  using const_iterator = ItemLocalIdListViewConstIteratorT<ItemType>;

 public:

  ItemLocalIdViewT() = default;

 private:

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdViewT(const Int32* ids, Int32 s, Int32 local_id_offset)
  : ItemLocalIdListView(ids, s, local_id_offset)
  {}

 public:

  ARCCORE_HOST_DEVICE LocalIdType operator[](Int32 i) const
  {
    return LocalIdType(localId(i));
  }

  constexpr ARCCORE_HOST_DEVICE const_iterator begin() const
  {
    return const_iterator(m_local_ids, m_local_id_offset);
  }
  constexpr ARCCORE_HOST_DEVICE const_iterator end() const
  {
    return const_iterator(m_local_ids + m_size, m_local_id_offset);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
