// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVectorView.cc                                           (C) 2000-2024 */
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
: m_index_view(local_ids)
{
  _init2(family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView::
ItemVectorView(IItemFamily* family, ItemIndexArrayView indexes)
: m_index_view(indexes)
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

void ItemVectorView::
fillLocalIds(Array<Int32>& ids) const
{
  m_index_view.fillLocalIds(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemIndexArrayView::
fillLocalIds(Array<Int32>& ids) const
{
  m_view.fillLocalIds(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Note: ces structures doivent avoir le même layout que la
// version qui est dans NumericWrapper.h

// Cette classe sert de type de retour pour wrapper la class ConstArrayView
template<typename DataType> class ConstArrayViewPOD_T
{
 public:
  Integer m_size;
  const DataType* m_ptr;
};

class ItemIndexArrayViewPOD
{
 public:
  ConstArrayViewPOD_T<Int32> m_local_ids;
  Int32 m_flags;
};

class ItemVectorViewPOD
{
 public:
  ItemIndexArrayViewPOD m_local_ids;
  ItemSharedInfo* m_shared_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemVectorView::
_internalSwigSet(ItemVectorViewPOD* vpod)
{
  vpod->m_local_ids.m_local_ids.m_size = localIds().size();
  vpod->m_local_ids.m_local_ids.m_ptr = localIds().unguardedBasePointer();
  vpod->m_local_ids.m_flags = indexes().flags();
  vpod->m_shared_info = m_shared_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

