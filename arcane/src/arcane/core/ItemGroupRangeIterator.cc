// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupRangeIterator.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Mesh entity groups.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/core/ItemGroupRangeIterator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemGroupImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupRangeIterator::
ItemGroupRangeIterator(const ItemGroup& group)
: m_current(0)
{
  if (group.null()) {
    m_kind = IK_Unknown;
    m_end = 0;
    m_items_local_ids = nullptr;
    return;
  }
  m_kind = group.itemKind();
  ItemGroupImpl* igi = group.internal();
  igi->checkNeedUpdate();
  Int32ConstArrayView local_ids(igi->itemsLocalId());
  m_end = local_ids.size();
  m_items_local_ids = local_ids.data();
  m_items = igi->itemInfoListView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupRangeIterator::
ItemGroupRangeIterator()
: m_kind(IK_Unknown)
, m_current(0)
, m_end(0)
, m_items_local_ids(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
