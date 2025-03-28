// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItem.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Entités matériau et milieux.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItem.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o, const ComponentCell& mvi)
{
  ConstituentItemLocalId x = mvi;
  o << x;
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentCell::
_badConversion(matimpl::ConstituentItemBase item_base, Int32 level,Int32 expected_level)
{
  ARCANE_FATAL("bad level for internal component cell level={0} expected={1} cid={2} component_id={3}",
               level,expected_level, item_base.m_constituent_item_index,item_base.componentId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
