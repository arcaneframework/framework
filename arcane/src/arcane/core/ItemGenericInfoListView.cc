// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGenericInfoListView.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Vue sur les informations génériques d'une famille d'entités.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGenericInfoListView.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGenericInfoListView::
ItemGenericInfoListView(ItemSharedInfo* shared_info)
: m_unique_ids(shared_info->m_unique_ids)
, m_owners(shared_info->m_owners)
, m_flags(shared_info->m_flags)
, m_type_ids(shared_info->m_type_ids)
, m_item_shared_info(shared_info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGenericInfoListView::
ItemGenericInfoListView(IItemFamily* family)
: ItemGenericInfoListView(family->itemInfoListView())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGenericInfoListView::
ItemGenericInfoListView(const ItemInfoListView& info_list_view)
: ItemGenericInfoListView(info_list_view.m_item_shared_info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
