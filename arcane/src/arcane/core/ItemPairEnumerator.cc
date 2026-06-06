// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairEnumerator.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Enumerator over an array of arrays of mesh entities.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemPairEnumerator.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairEnumerator::
ItemPairEnumerator(const ItemPairGroup& array)
: m_current(0)
, m_indexes(array.internal()->indexes())
, m_items_local_id(array.itemGroup().internal()->itemsLocalId())
, m_sub_items_local_id(array.internal()->subItemsLocalId())
{
  m_end = m_indexes.size() - 1;

  ItemInfoListView items_view(array.internal()->itemFamily());
  ItemInfoListView sub_items_view(array.internal()->subItemFamily());

  m_items_shared_info = items_view.m_item_shared_info;
  m_sub_items_shared_info = sub_items_view.m_item_shared_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
