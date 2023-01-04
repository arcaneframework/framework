// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVector.cc                                               (C) 2000-2023 */
/*                                                                           */
/* Vecteur (tableau indirect) d'entités.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemVector.h"

#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector(IItemFamily* afamily)
: m_items(afamily->itemsInternal())
, m_family(afamily)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector(IItemFamily* afamily, Int32ConstArrayView local_ids)
: m_items(afamily->itemsInternal())
, m_local_ids(local_ids)
, m_family(afamily)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector(IItemFamily* afamily, Integer asize)
: m_items(afamily->itemsInternal())
, m_family(afamily)
{
  m_local_ids.resize(asize);
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemVector::
_init()
{
  if (m_family) {
    ItemInfoListView info_view(m_family);
    m_shared_info = info_view.m_item_shared_info;
  }
  else
    m_shared_info = ItemSharedInfo::nullInstance();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemVector::
setFamily(IItemFamily* afamily)
{
  m_local_ids.clear();
  m_family = afamily;
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ItemVectorViewT<Node>;
template class ItemVectorViewT<Edge>;
template class ItemVectorViewT<Face>;
template class ItemVectorViewT<Cell>;
template class ItemVectorViewT<Particle>;
template class ItemVectorViewT<DoF>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
