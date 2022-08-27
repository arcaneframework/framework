// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVectorView.cc                                           (C) 2000-2022 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"

#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView::
ItemVectorView(IItemFamily* family, ConstArrayView<Int32> local_ids)
: m_items(family->itemsInternal())
, m_local_ids(local_ids)
{
  _init2(family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView::
ItemVectorView(IItemFamily* family, ItemIndexArrayView indexes)
: m_items(family->itemsInternal())
, m_local_ids(indexes)
{
 _init2(family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemVectorView::
_init2(IItemFamily* family)
{
  if (family){
    ItemInfoListView info_view(family);
    m_shared_info = info_view.m_item_shared_info;
  }
  else
    m_shared_info = ItemSharedInfo::nullInstance();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

