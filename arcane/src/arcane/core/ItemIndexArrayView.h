// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemIndexArrayView.h                                        (C) 2000-2025 */
/*                                                                           */
/* View over an index array (localIds()) of entities.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMINDEXARRAYVIEW_H
#define ARCANE_CORE_ITEMINDEXARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief View over an index array (localIds()) of entities.
 *
 * \warning The view is only valid as long as the associated array is not
 * modified. Instances of this class are generally temporary
 * and should not be retained.
 *
 * In addition to the list of entities, this class allows for
 * additional information, such as whether the list is contiguous.
 */
class ARCANE_CORE_EXPORT ItemIndexArrayView
{
  // NOTE: This class is mapped in C# and if its structure is changed, it
  // must update the corresponding C# version.
  friend ItemVectorView;
  friend ItemGroup;
  template <int Extent> friend class ItemConnectedListView;
  template <typename ItemType, int Extent> friend class ItemConnectedListViewT;
  template <typename ItemType> friend class ItemVectorViewT;

 public:

  // NOTE: If values are added here, it must be checked whether they should be
  // propagated in methods such as subView().
  enum
  {
    F_Contiguous = 1 << 1, //!< The local IDs are contiguous.
    F_Contigous = F_Contiguous
  };

 public:

  //! Constructs an empty view
  ItemIndexArrayView() = default;

  // TODO: To be removed
  //! Constructs a view from the local IDs \a local_ids
  explicit ItemIndexArrayView(const Int32ConstArrayView local_ids)
  : m_view(local_ids, 0)
  {}

  explicit ItemIndexArrayView(const impl::ItemLocalIdListContainerView& view)
  : m_view(view)
  {
  }

 public:

  //! Accesses the i-th element of the vector
  inline Int32 operator[](Integer index) const
  {
    return m_view.localId(index);
  }

  //! Number of elements in the vector
  Int32 size() const
  {
    return m_view.size();
  }

  //! Adds the list of the vector's localIds() to \a ids.
  void fillLocalIds(Array<Int32>& ids) const;

  //! Sub-view starting from element \a abegin and containing \a asize elements
  inline ItemIndexArrayView subView(Integer abegin, Integer asize) const
  {
    // The F_Contiguous flag is propagated to the sub-view.
    // For other flags, it will be necessary to check if they should be propagated.
    return ItemIndexArrayView(m_view._idsWithoutOffset().subView(abegin, asize), m_view.m_local_id_offset, m_flags);
  }

  Int32 flags() const
  {
    return m_flags;
  }

  bool isContigous() const { return isContiguous(); }

  //! True if the localIds() are contiguous
  bool isContiguous() const
  {
    return m_flags & F_Contigous;
  }

  friend std::ostream& operator<<(std::ostream& o, const ItemIndexArrayView& a)
  {
    o << a.m_view;
    return o;
  }

 public:

  // TODO Deprecate (3.11+)
  //! Array of local IDs of entities
  Int32ConstArrayView localIds() const
  {
    return m_view._idsWithoutOffset();
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  operator Int32ConstArrayView() const
  {
    return _localIds();
  }

 private:

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  const Int32* unguardedBasePointer() const
  {
    return _data();
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  const Int32* data() const
  {
    return _data();
  }

 protected:

  impl::ItemLocalIdListContainerView m_view;
  Int32 m_flags = 0;

 private:

  ItemIndexArrayView(SmallSpan<const Int32> local_ids, Int32 local_id_offset, Int32 aflags)
  : m_view(local_ids, local_id_offset)
  , m_flags(aflags)
  {}

  const Int32* _data() const
  {
    return m_view.m_local_ids;
  }

  Int32ConstArrayView _localIds() const
  {
    return m_view._idsWithoutOffset();
  }
  Int32 _localIdOffset() const
  {
    return m_view.m_local_id_offset;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
