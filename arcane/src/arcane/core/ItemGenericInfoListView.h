// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGenericInfoListView.h                                   (C) 2000-2025 */
/*                                                                           */
/* View of the generic information of an entity family.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMGENERICINFOLISTVIEW_H
#define ARCANE_CORE_ITEMGENERICINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemUniqueId.h"
#include "arcane/core/ItemLocalId.h"
#include "arcane/core/ItemFlags.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of the generic information of an entity family.
 *
 * Like all views, instances of this class are temporary
 * and should not be kept when the associated family is modified.
 */
class ARCANE_CORE_EXPORT ItemGenericInfoListView
{
  friend class ItemInfoListView;

 public:

  ItemGenericInfoListView() = default;

  /*!
   * \brief Constructs a view associated with the family \a family.
   *
   * \a family may be \a nullptr in which case the instance is
   * not usable for retrieving information about entities
   */
  explicit ItemGenericInfoListView(IItemFamily* family);

  explicit ItemGenericInfoListView(const ItemInfoListView& info_list_view);

 public:

  //! Associated family
  IItemFamily* itemFamily() const { return m_item_shared_info->itemFamily(); }

  //! Owner of the entity with local ID \a local_id
  constexpr ARCCORE_HOST_DEVICE Int32 owner(Int32 local_id) const { return m_owners[local_id]; }

  //! Owner of the entity with local ID \a local_id
  constexpr ARCCORE_HOST_DEVICE Int32 owner(ItemLocalId local_id) const { return m_owners[local_id.localId()]; }

  //! Type of the entity with local ID \a local_id
  constexpr ARCCORE_HOST_DEVICE Int16 typeId(Int32 local_id) const { return m_type_ids[local_id]; }

  //! Type of the entity with local ID \a local_id
  constexpr ARCCORE_HOST_DEVICE Int16 typeId(ItemLocalId local_id) const { return m_type_ids[local_id.localId()]; }

  //! uniqueId() of the entity with local ID \a local_id
  ARCCORE_HOST_DEVICE ItemUniqueId uniqueId(Int32 local_id) const
  {
    return ItemUniqueId{ m_unique_ids[local_id] };
  }

  //! uniqueId() of the entity with local ID \a local_id
  ARCCORE_HOST_DEVICE ItemUniqueId uniqueId(ItemLocalId local_id) const
  {
    return ItemUniqueId{ m_unique_ids[local_id.localId()] };
  }

  //! Indicates if the entity with local ID \a local_id belongs to the subdomain
  constexpr ARCCORE_HOST_DEVICE bool isOwn(Int32 local_id) const
  {
    return ItemFlags::isOwn(m_flags[local_id]);
  }

  //! Indicates if the entity with local ID \a local_id belongs to the subdomain
  constexpr ARCCORE_HOST_DEVICE bool isOwn(ItemLocalId local_id) const
  {
    return ItemFlags::isOwn(m_flags[local_id]);
  }

  //! Indicates if the entity with local ID \a local_id is shared by other subdomains
  constexpr ARCCORE_HOST_DEVICE bool isShared(Int32 local_id) const
  {
    return ItemFlags::isShared(m_flags[local_id]);
  }

  //! Indicates if the entity with local ID \a local_id is shared by other subdomains
  constexpr ARCCORE_HOST_DEVICE bool isShared(ItemLocalId local_id) const
  {
    return ItemFlags::isShared(m_flags[local_id]);
  }

 private:

  // NOTE: This structure is used in the C# wrapping.
  // If the fields are modified, the equivalent C# structure must be updated
  Int64ArrayView m_unique_ids;
  Int32ArrayView m_owners;
  Int32ArrayView m_flags;
  Int16ArrayView m_type_ids;
  ItemSharedInfo* m_item_shared_info = ItemSharedInfo::nullInstance();

 private:

  // Only ItemFamily can create instances via this constructor
  explicit ItemGenericInfoListView(ItemSharedInfo* shared_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
