﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairEnumerator.h                                        (C) 2000-2021 */
/*                                                                           */
/* Enumérateur sur un tableau de tableau d'entités du maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMPAIRENUMERATOR_H
#define ARCANE_ITEMPAIRENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternalEnumerator.h"
#include "arcane/Item.h"

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
  ItemPairEnumerator();

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
  inline ItemInternalEnumerator subItems() const
  {
    return { m_sub_items_internal.data(),
             m_sub_items_local_id.data() + m_indexes[m_current],
             static_cast<Int32>(m_indexes[m_current + 1] - m_indexes[m_current]) };
  }
  inline Item operator*() const
  {
    return Item(m_items_internal.data(), m_items_local_id[m_current]);
  }
  inline Integer nbSubItem() const
  {
    return static_cast<Int32>(m_indexes[m_current + 1] - m_indexes[m_current]);
  }

 protected:
  Int32 m_current;
  Int32 m_end;
  Int64ConstArrayView m_indexes;
  Int32ConstArrayView m_items_local_id;
  Span<const Int32> m_sub_items_local_id;
  ItemInternalList m_items_internal;
  ItemInternalList m_sub_items_internal;
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
    return ItemType(m_items_internal.data(), m_items_local_id[m_current]);
  }
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
 public:
  ItemPairEnumeratorT(const ItemPairGroupT<ItemType, SubItemType>& array)
  : ItemPairEnumeratorSubT<ItemType>(array)
  {
  }
  inline ItemEnumeratorT<SubItemType> subItems() const
  {
    return { this->m_sub_items_internal.data(),
             this->m_sub_items_local_id.data() + this->m_indexes[this->m_current],
             static_cast<Int32>(this->m_indexes[this->m_current + 1] - this->m_indexes[this->m_current]) };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
