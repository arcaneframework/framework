// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.cc                                         (C) 2000-2024 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInfoListView.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInfoListView::
ItemInfoListView(IItemFamily* family)
: ItemInfoListView(family ? family->_internalApi()->commonItemSharedInfo() : ItemSharedInfo::nullInstance())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que le genre d'entité correspond à celui attendu.
 */
void ItemInfoListView::
_checkValid(eItemKind expected_kind)
{
  IItemFamily* family = itemFamily();
  if (!family)
    return;
  eItemKind my_kind = family->itemKind();
  if (my_kind != expected_kind)
    ARCANE_FATAL("Bad kind family={0} kind={1} expected_kind={2}",
                 family->fullName(), my_kind, expected_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
