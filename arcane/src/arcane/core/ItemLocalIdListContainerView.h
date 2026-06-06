// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalIdListContainerView.h                              (C) 2000-2024 */
/*                                                                           */
/* View over the container of a list of ItemLocalId.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMLOCALIDLISTCONTAINERVIEW_H
#define ARCANE_CORE_ITEMLOCALIDLISTCONTAINERVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief View over the container of a list of ItemLocalId.
 *
 * This class is only used to store the necessary information
 * for a list of 'ItemLocalId' and is only used to pass
 * information between entity views (e.g., ItemVectorView)
 * and associated iterators.
 *
 * The associated user class is ItemLocalIdListView.
 */
class ARCANE_CORE_EXPORT ItemLocalIdListContainerView
{
  // NOTE: This class is mapped in C# and if its structure is changed,
  // the corresponding C# version must be updated.
  template <typename ItemType> friend class ::Arcane::ItemLocalIdListViewT;
  template <int Extent> friend class ::Arcane::impl::ItemIndexedListView;
  friend ItemVectorView;
  friend ItemLocalIdListView;
  friend ItemInternalConnectivityList;
  friend ItemInternalVectorView;
  friend ItemEnumeratorBase;
  friend SimdItemEnumeratorBase;
  friend ItemIndexArrayView;

 private:

  ItemLocalIdListContainerView() = default;
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListContainerView(const Int32* ids, Int32 s, Int32 local_id_offset)
  : m_local_ids(ids)
  , m_local_id_offset(local_id_offset)
  , m_size(s)
  {}

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListContainerView(SmallSpan<const Int32> ids, Int32 local_id_offset)
  : m_local_ids(ids.data())
  , m_local_id_offset(local_id_offset)
  , m_size(ids.size())
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 operator[](Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_size);
    return m_local_ids[index] + m_local_id_offset;
  }
  constexpr ARCCORE_HOST_DEVICE Int32 localId(Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_size);
    return m_local_ids[index] + m_local_id_offset;
  }
  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_size; }

  void fillLocalIds(Array<Int32>& ids) const;

  friend ARCANE_CORE_EXPORT std::ostream&
  operator<<(std::ostream& o, const ItemLocalIdListContainerView& lhs);

 private:

  ConstArrayView<Int32> _idsWithoutOffset() const { return { m_size, m_local_ids }; }

 private:

  const Int32* m_local_ids = nullptr;
  Int32 m_local_id_offset = 0;
  Int32 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
