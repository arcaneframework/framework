// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemAllocationInfo                                          (C) 2000-2023 */
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
  };

  struct FamilyInfo
  {
    String name;
    eItemKind item_kind;
    Int64ConstSmallSpan item_uids;
    UniqueArray<ConnectedFamilyInfo> connected_family_info;
    Real3ConstSmallSpan item_coordinates; // if needed
    String item_coordinates_variable_name; // if needed
  };
  UniqueArray<FamilyInfo> family_infos;
};

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_ITEMALLOCATIONINFO_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
