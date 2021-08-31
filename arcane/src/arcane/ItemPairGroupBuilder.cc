// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupBuilder.cc                                     (C) 2000-2021 */
/*                                                                           */
/* Construction des listes des entités des ItemPairGroup.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/ItemPairGroupBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupBuilder::
ItemPairGroupBuilder(const ItemPairGroup& group)
: m_group(group)
, m_index(0)
, m_unguarded_indexes(m_group.internal()->unguardedIndexes())
, m_unguarded_local_ids(m_group.internal()->unguardedLocalIds())
{
  m_unguarded_indexes.clear();
  m_unguarded_local_ids.clear();
  m_unguarded_indexes.add(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupBuilder::
~ItemPairGroupBuilder()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupBuilder::
addNextItem(Int32ConstArrayView sub_items)
{
  ++m_index;
  m_unguarded_local_ids.addRange(sub_items);
  m_unguarded_indexes.add(m_unguarded_local_ids.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
