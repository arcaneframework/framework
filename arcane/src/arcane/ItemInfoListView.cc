// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInfoListView.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInfoListView::
ItemInfoListView(IItemFamily* family)
{
  if (family)
    *this = family->itemInfoListView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que le genre d'entité correspond à celui attendu.
 */
void ItemInfoListView::
_checkValid(eItemKind expected_kind)
{
  if (!m_family)
    return;
  eItemKind my_kind = m_family->itemKind();
  if (my_kind != expected_kind)
    ARCANE_FATAL("Bad kind family={0} kind={1} expected_kind={2}",
                 m_family->fullName(), my_kind, expected_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
