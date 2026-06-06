// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemIndexedListView.h                                       (C) 2000-2023 */
/*                                                                           */
/* View of an indexed list of entities.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINDEXEDLISTVIEW_H
#define ARCANE_ITEMINDEXEDLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemLocalIdListContainerView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal view of an array of entities.
 *
 * This class is only useful for building entity lists used internally by
 * %Arcane. The user version of this class is ItemConnectivityView. The main
 * difference between the two classes is that this one only maintains the
 * lists but does not allow, for example, returning a typed entity.
 *
 * \sa ItemConnectedListView
 */
template <int Extent>
class ARCANE_CORE_EXPORT ItemIndexedListView
{
  static_assert(Extent == (-1), "only dynamic (-1) extent is currently supported");

  friend ItemInternalConnectivityList;
  friend ItemBase;
  friend ItemVectorView;
  friend ItemInternalVectorView;
  friend class ItemConnectedListView<DynExtent>;
  template <typename T> friend class Arcane::ItemEnumeratorBaseT;
  template <typename T> friend class Arcane::ItemConnectedEnumeratorBaseT;

 public:

  ItemIndexedListView() = default;

 private:

  // TODO: To be removed
  constexpr ItemIndexedListView(ItemSharedInfo* si, SmallSpan<const Int32> local_ids, Int32 local_id_offset)
  : m_local_ids(local_ids)
  , m_shared_info(si)
  , m_local_id_offset(local_id_offset)
  {
    ARCANE_ASSERT(m_shared_info, ("null shared_info"));
  }

  constexpr ItemIndexedListView(ItemSharedInfo* si, const impl::ItemLocalIdListContainerView& container_view)
  : m_local_ids(container_view.m_local_ids, container_view.m_size)
  , m_shared_info(si)
  , m_local_id_offset(container_view.m_local_id_offset)
  {
    ARCANE_ASSERT(m_shared_info, ("null shared info"));
  }

 private:

  //! Number of elements in the vector
  constexpr Int32 size() const { return m_local_ids.size(); }

  //! Array of local entity IDs
  constexpr SmallSpan<const Int32, Extent> localIds() const { return m_local_ids; }

  //! Array of local entity IDs
  ConstArrayView<Int32> constLocalIds() const { return m_local_ids.constSmallView(); }

 private:

  SmallSpan<const Int32, Extent> m_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
  Int32 m_local_id_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
