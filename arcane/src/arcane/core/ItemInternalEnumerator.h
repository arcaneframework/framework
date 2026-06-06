// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalEnumerator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Enumerator over a list of ItemInternal.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMINTERNALENUMERATOR_H
#define ARCANE_CORE_ITEMINTERNALENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternalVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;
class ItemEnumerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Enumerator over a list of entities.
 * \deprecated This class is obsolete and should no longer be used. You
 * must use ItemEnumerator instead.
 */
class ItemInternalEnumerator
{
 private:

  typedef ItemInternal* ItemInternalPtr;
  friend class ItemEnumerator;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This class is deprecated. Use ItemEnumerator instead")
  ItemInternalEnumerator(const ItemInternalPtr* items, const Int32* local_ids, Integer n)
  : m_items(items)
  , m_local_ids(local_ids)
  , m_index(0)
  , m_count(n)
  {
  }

  ARCANE_DEPRECATED_REASON("Y2022: This class is deprecated. Use ItemEnumerator instead")
  ItemInternalEnumerator(const ItemInternalVectorView& view)
  : m_items(view._items().data())
  , m_local_ids(view.localIds().data())
  , m_index(0)
  , m_count(view.size())
  {}

  ARCANE_DEPRECATED_REASON("Y2022: This class is deprecated. Use ItemEnumerator instead")
  ItemInternalEnumerator(const ItemInternalArrayView& items, const Int32ConstArrayView& local_ids)
  : m_items(items.data())
  , m_local_ids(local_ids.data())
  , m_index(0)
  , m_count(local_ids.size())
  {}

 public:

  ItemInternal* operator*() const { return m_items[m_local_ids[m_index]]; }
  ItemInternal* operator->() const { return m_items[m_local_ids[m_index]]; }
  inline void operator++() { ++m_index; }
  inline bool operator()() { return m_index < m_count; }
  inline bool hasNext() { return m_index < m_count; }

  //! Number of elements in the enumerator
  inline Integer count() const { return m_count; }

  //! Current index of the enumerator
  inline Integer index() const { return m_index; }

  //! localId() of the current entity.
  inline Integer localId() const { return m_local_ids[m_index]; }

 protected:

  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;
  Integer m_count;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
