// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupRangeIterator.cc                                   (C) 2000-2018 */
/*                                                                           */
/* Groupes d'entités du maillage.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ItemGroupRangeIterator.h"
#include "arcane/ItemGroup.h"
#include "arcane/ItemGroupImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupRangeIterator::
ItemGroupRangeIterator(const ItemGroup& group)
: m_current(0)
{
  if (group.null()){
    m_end = 0;
    m_items_local_ids = 0;
    m_items = 0;
    return;
  }
  m_kind = group.itemKind();
  ItemGroupImpl* igi = group.internal();
  igi->checkNeedUpdate();
  Int32ConstArrayView local_ids(igi->itemsLocalId());
  m_end = local_ids.size();
  m_items_local_ids = local_ids.data();
  m_items = igi->itemsInternal().data();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupRangeIterator::
ItemGroupRangeIterator()
: m_kind(IK_Unknown)
, m_current(0)
, m_end(0)
, m_items_local_ids(0)
, m_items(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
