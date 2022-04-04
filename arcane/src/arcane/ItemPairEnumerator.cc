// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairEnumerator.cc                                       (C) 2000-2007 */
/*                                                                           */
/* Enumérateur sur un tableau de tableau d'entités du maillage.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ItemPairEnumerator.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairEnumerator::
ItemPairEnumerator(const ItemPairGroup& array)
: m_current(0)
, m_indexes(array.internal()->indexes())
, m_items_local_id(array.itemGroup().internal()->itemsLocalId())
, m_sub_items_local_id(array.internal()->subItemsLocalId())
, m_items_internal(array.internal()->itemFamily()->itemsInternal())
, m_sub_items_internal(array.internal()->subItemFamily()->itemsInternal())
{
  m_end = m_indexes.size()-1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairEnumerator::
ItemPairEnumerator()
: m_current(0)
, m_end(0)
, m_indexes()
, m_items_local_id()
, m_sub_items_local_id()
, m_items_internal()
, m_sub_items_internal()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
