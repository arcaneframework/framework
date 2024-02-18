// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternal.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Partie interne d'une maille matériau ou milieu.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemInternal.h"

#include "arcane/utils/BadCastException.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemInternal ComponentItemInternal::nullComponentItemInternal;

ComponentItemSharedInfo ComponentItemSharedInfo::null_shared_info;
ComponentItemSharedInfo* ComponentItemSharedInfo::null_shared_info_pointer = &ComponentItemSharedInfo::null_shared_info;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o,const ComponentItemInternalLocalId& id)
{
  o << id.localId();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemLocalIdListView::
_checkCoherency() const
{
  Int32 nb_item = m_items_internal.size();
  if (!m_component_shared_info)
    ARCANE_FATAL("Null ComponentItemSharedInfo nb_item={0}", nb_item);
  return;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemLocalIdListView::
_throwIncoherentSharedInfo(Int32 index) const
{
  ARCANE_FATAL("Incoherent ComponentItemSharedInfo for item index={0}", index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
