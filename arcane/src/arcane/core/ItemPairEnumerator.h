// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairEnumerator.h                                        (C) 2000-2023 */
/*                                                                           */
/* Enumérateur sur un tableau de tableau d'entités du maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMPAIRENUMERATOR_H
#define ARCANE_ITEMPAIRENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;
class ItemItemArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur un tableau de tableaux d'entités du maillage.
 */
class ARCANE_CORE_EXPORT ItemPairEnumerator
{
 public:

  typedef ItemInternal* ItemInternalPtr;

 public:

  ItemPairEnumerator(const ItemPairGroup& array);
  ItemPairEnumerator() = default;

 public:

  inline void operator++()
  {
    ++m_current;
  }
  inline bool hasNext() const
  {
    return m_current < m_end;
  }
  inline Int32 itemLocalId() const
  {
    return m_items_local_id[m_current];
  }
  inline Int32 index() const
  {
    return m_current;
  }
  inline ItemEnumerator subItems() const
  {
    return { m_sub_items_shared_info, _ids() };
  }
  inline Item operator*() const
  {
    return Item(m_items_local_id[m_current], m_items_shared_info);
  }
  inline Integer nbSubItem() const
  {
    return static_cast<Int32>(m_indexes[m_current + 1] - m_indexes[m_current]);
  }

  //! Conversion vers un ItemLocalIdT<ItemType>
  operator ItemLocalId() const { return ItemLocalId{ itemLocalId() }; }

 protected:

  Int32 m_current = 0;
  Int32 m_end = 0;
  Int64ConstArrayView m_indexes;
  Int32ConstArrayView m_items_local_id;
  Span<const Int32> m_sub_items_local_id;
  ItemSharedInfo* m_items_shared_info = ItemSharedInfo::nullInstance();
  ItemSharedInfo* m_sub_items_shared_info = ItemSharedInfo::nullInstance();

 protected:

  Item _currentItem() const
  {
    return Item(m_items_local_id[m_current], m_items_shared_info);
  }
  ConstArrayView<Int32> _ids() const
  {
    return ConstArrayView<Int32>(nbSubItem(), m_sub_items_local_id.data() + m_indexes[m_current]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur un tableau de tableaux d'entités
 * du maillage de genre \a ItemType et \a SubItemType.
 */
template <typename ItemType>
class ItemPairEnumeratorSubT
: public ItemPairEnumerator
{
 public:

  ItemPairEnumeratorSubT(const ItemPairGroup& array)
  : ItemPairEnumerator(array)
  {
  }

 public:

  inline ItemType operator*() const
  {
    return ItemType(this->_currentItem());
  }
  //! Conversion vers un ItemLocalIdT<ItemType>
  operator ItemLocalIdT<ItemType>() const { return ItemLocalIdT<ItemType>{ this->itemLocalId() }; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur un tableau de tableaux d'entités
 * du maillage de genre \a ItemType et \a SubItemType.
 */
template <typename ItemType, typename SubItemType>
class ItemPairEnumeratorT
: public ItemPairEnumeratorSubT<ItemType>
{
  using BaseClass = ItemPairEnumeratorSubT<ItemType>;

 public:

  ItemPairEnumeratorT(const ItemPairGroupT<ItemType, SubItemType>& array)
  : ItemPairEnumeratorSubT<ItemType>(array)
  {
  }

  ItemEnumeratorT<SubItemType> subItems() const
  {
    return { this->m_sub_items_shared_info, this->_ids() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
