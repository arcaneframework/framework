// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternal.cc                                    (C) 2000-2023 */
/*                                                                           */
/* Partie interne d'une maille matériau ou milieu.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemInternal.h"

#include "arcane/utils/BadCastException.h"

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

void ComponentItemInternal::
_throwBadCast(Int32 v)
{
  throw BadCastException(A_FUNCINFO,String::format("Can not cast v={0} to type 'Int16'",v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
