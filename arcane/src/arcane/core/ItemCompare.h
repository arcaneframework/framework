// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemCompare.h                                               (C) 2000-2025 */
/*                                                                           */
/* Routines for comparing two entities.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMCOMPARE_H
#define ARCANE_CORE_ITEMCOMPARE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemCompare
{
 public:

  bool operator()(const Item& item1, const Item& item2) const
  {
    return item1.uniqueId() < item2.uniqueId();
  }
  bool operator()(const ItemInternal* item1, const ItemInternal* item2) const
  {
    return item1->uniqueId() < item2->uniqueId();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * Specific order for Item* to be used in a set.
 * We use the ID, but the primary sort is by level.
 * This ensures, when traversing the set from beginning to end,
 * that lower level elements (parents) are encountered first.
 */
class CompareItemIdsByLevel
{
 public:

  bool operator()(const Cell& item1, const Cell& item2) const
  {
    const Integer i1_l = item1.level(), i2_l = item2.level();
    Int64 i1_id = item1.uniqueId();
    Int64 i2_id = item2.uniqueId();

    return (i1_l == i2_l) ? i1_id < i2_id : i1_l < i2_l;
  }

  bool operator()(const ItemInternal* item1, const ItemInternal* item2) const
  {
    ARCANE_CHECK_PTR(item1);
    ARCANE_CHECK_PTR(item2);
    const Integer i1_l = item1->level(), i2_l = item2->level();
    Int64 i1_id = item1->uniqueId();
    Int64 i2_id = item2->uniqueId();

    return (i1_l == i2_l) ? i1_id < i2_id : i1_l < i2_l;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
