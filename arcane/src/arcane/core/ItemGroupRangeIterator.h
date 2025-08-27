// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupRangeIterator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Intervalle d'itération sur les entités d'un groupe du maillage.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMGROUPRANGEITERATOR_H
#define ARCANE_CORE_ITEMGROUPRANGEITERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Iterateur sur les éléments d'un groupe.
 */
class ARCANE_CORE_EXPORT ItemGroupRangeIterator
{
 public:

  typedef ItemInternal* ItemPtr;
  typedef ItemPtr* Iterator;

  ItemGroupRangeIterator(const ItemGroup& group);
  ItemGroupRangeIterator();

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
    return m_items_local_ids[m_current];
  }
  inline Integer index() const
  {
    return m_current;
  }
  inline eItemKind kind() const
  {
    return m_kind;
  }

 protected:

  eItemKind m_kind;
  Integer m_current;
  Integer m_end;
  const Int32* ARCANE_RESTRICT m_items_local_ids;
  ItemInfoListView m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération sur un groupe d'entités de maillage.
 */
template <typename T>
class ItemGroupRangeIteratorT
: public ItemGroupRangeIterator
{
 public:

  inline ItemGroupRangeIteratorT(const ItemGroup& group)
  : ItemGroupRangeIterator(group)
  {
  }
  inline ItemGroupRangeIteratorT()
  : ItemGroupRangeIterator()
  {
  }

 public:

  T operator*() const
  {
    return T(m_items[m_items_local_ids[m_current]]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
