// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DualUniqueArray.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Tableau 1D alloué à la fois sur CPU et accélérateur.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/DualUniqueArray.h"

#include "arcane/utils/String.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DualUniqueArrayBase::
_memoryCopy(Span<const std::byte> from, Span<std::byte> to)
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  eMemoryRessource from_mem = eMemoryRessource::Unknown;
  eMemoryRessource to_mem = eMemoryRessource::Unknown;
  mrm->_internal()->copy(ConstMemoryView(from), from_mem, MutableMemoryView(to), to_mem, nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
