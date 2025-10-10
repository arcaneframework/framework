// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemAllocationInfo                                          (C) 2000-2025 */
/*                                                                           */
/* AllocationInfo for mesh using eItemAllocationInfo mode                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMALLOCATIONINFO_H
#define ARCANE_ITEMALLOCATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemAllocationInfo
{
 public:

  struct ConnectedFamilyInfo
  {
    String name;
    eItemKind item_kind;
    String connectivity_name;
    Int32ConstSmallSpan nb_connected_items_per_item;
    Int64ConstSmallSpan connected_items_uids;
    // If ItemAllocation has to own the data use the followings.
    // In this case, Store a view in the corresponding spans using method updateViewsFromInternalData().
    Int32UniqueArray _nb_connected_items_per_item_data;
    Int64UniqueArray _connected_items_uids_data;

    void updateViewsFromInternalData()
    {
      nb_connected_items_per_item = _nb_connected_items_per_item_data.constSmallSpan();
      connected_items_uids = _connected_items_uids_data.constSmallSpan();
    }
  };

  struct FamilyInfo
  {
    String name;
    eItemKind item_kind;
    Int64ConstSmallSpan item_uids;
    UniqueArray<ConnectedFamilyInfo> connected_family_infos;
    Real3ConstSmallSpan item_coordinates; // if needed
    String item_coordinates_variable_name; // if needed
    Int32UniqueArray item_owners; // if needed
    // If ItemAllocation has to own the data use the followings.
    // In this case, Store a view in the corresponding spans using method updateViewsFromInternalData().
    Int64UniqueArray _item_uids_data;
    Real3UniqueArray _item_coordinates_data;

    void updateViewsFromInternalData()
    {
      item_uids = _item_uids_data.constSmallSpan();
      item_coordinates = _item_coordinates_data.constSmallSpan();
      for (auto& connected_family_info : connected_family_infos) {
        connected_family_info.updateViewsFromInternalData();
      }
    }

    void clear()
    {
      name = String{};
      item_kind = IK_Unknown;
      item_uids = Int64ConstSmallSpan{};
      connected_family_infos.clear();
      item_coordinates = Real3ConstSmallSpan{};
      item_coordinates_variable_name = String{};
      _item_uids_data.clear();
      _item_coordinates_data.clear();
      item_owners.clear();
    }
  };
  UniqueArray<FamilyInfo> family_infos;

  // To use if ItemAllocationInfo has to own the data. The views will point to the stored data
  void updateViewsFromInternalData()
  {
    for (auto& family_info : family_infos) {
     family_info.updateViewsFromInternalData();
    }
  }

  void clear()
  {
    family_infos.clear();
  }
};

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_ITEMALLOCATIONINFO_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
