// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemArrayEnumerator.h                                       (C) 2000-2025 */
/*                                                                           */
/* Énumérateur sur un tableau d'entités du maillage.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMARRAYENUMERATOR_H
#define ARCANE_CORE_ITEMARRAYENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Énumérateur sur un tableau d'entités du maillage.
 */
class ARCANE_CORE_EXPORT ItemArrayEnumerator
{
 public:

  typedef ItemInternal* ItemPtr;
  typedef ItemPtr* Iterator;

 public:

  ItemArrayEnumerator(const Int32ConstArrayView ids, const ItemInternalList& items_internal)
  : m_current(0)
  , m_end(ids.size())
  , m_items_local_id(ids.data())
  , m_items_internal(items_internal.data())
  {
  }
  ItemArrayEnumerator(const Int32* ids, Integer nb_item, const ItemPtr* items_internal)
  : m_current(0)
  , m_end(nb_item)
  , m_items_local_id(ids)
  , m_items_internal(items_internal)
  {
  }

 public:

  inline void operator++()
  {
    ++m_current;
  }
  inline bool hasNext() const
  {
    return m_current < m_end;
  }
  inline Integer itemLocalId() const
  {
    return m_items_local_id[m_current];
  }
  inline Integer index() const
  {
    return m_current;
  }
  inline Item operator*() const
  {
    return Item(m_items_internal, m_items_local_id[m_current]);
  }

 protected:

  Integer m_current;
  Integer m_end;
  const Int32* ARCANE_RESTRICT m_items_local_id;
  const ItemPtr* m_items_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Énumérateur sur un tableau d'entités du maillage de genre \a ItemType.
 */
template <typename ItemType>
class ItemArrayEnumeratorT
: public ItemArrayEnumerator
{
 public:

  ItemArrayEnumeratorT(const Int32ConstArrayView ids, const ItemInternalList& items_internal)
  : ItemArrayEnumerator(ids, items_internal)
  {
  }
  ItemArrayEnumeratorT(const Int32* ids, Integer nb_item, const ItemPtr* items_internal)
  : ItemArrayEnumerator(ids, nb_item, items_internal)
  {
  }
  inline ItemType operator*() const
  {
    return ItemType(m_items_internal, m_items_local_id[m_current]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
