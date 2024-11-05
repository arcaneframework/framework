// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVector.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Vecteur d'entités de même genre.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemVector.h"

#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace
{
IMemoryAllocator* _getAllocator()
{
  return MemoryUtils::getDefaultDataAllocator();
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector(IItemFamily* afamily)
: m_local_ids(_getAllocator())
, m_family(afamily)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector(IItemFamily* afamily, Int32ConstArrayView local_ids)
: m_local_ids(_getAllocator())
, m_family(afamily)
{
  _init();
  m_local_ids.resize(local_ids.size());
  MemoryUtils::copy(m_local_ids.span(),Span<const Int32>(local_ids));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector(IItemFamily* afamily, Integer asize)
: m_local_ids(_getAllocator())
, m_family(afamily)
{
  m_local_ids.resize(asize);
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVector::
ItemVector()
: m_local_ids(_getAllocator())
{
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

template class ItemVectorT<Node>;
template class ItemVectorT<Edge>;
template class ItemVectorT<Face>;
template class ItemVectorT<Cell>;
template class ItemVectorT<Particle>;
template class ItemVectorT<DoF>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
