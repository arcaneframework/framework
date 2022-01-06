// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.cc                                                 (C) 2000-2022 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques sur accélérateur.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* NumArrayBaseCommon::
_getDefaultAllocator()
{
  return _getDefaultAllocator(eMemoryRessource::UnifiedMemory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* NumArrayBaseCommon::
_getDefaultAllocator(eMemoryRessource r)
{
  return platform::getDataMemoryRessourceMng()->getAllocator(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayBaseCommon::
_checkHost(eMemoryRessource r)
{
  if (r == eMemoryRessource::Host || r == eMemoryRessource::UnifiedMemory)
    return;
  ARCANE_FATAL("Invalid access from '{0}' ressource memory to host memory");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayBaseCommon::
_copy(Span<const std::byte> from, eMemoryRessource from_mem,
      Span<std::byte> to, eMemoryRessource to_mem)
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  mrm->_internal()->copy(from, from_mem, to, to_mem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class NumArray<Real, 4>;
template class NumArray<Real, 3>;
template class NumArray<Real, 2>;
template class NumArray<Real, 1>;

template class NumArrayBase<Real, 4>;
template class NumArrayBase<Real, 3>;
template class NumArrayBase<Real, 2>;
template class NumArrayBase<Real, 1>;

template class ArrayStridesBase<1>;
template class ArrayStridesBase<2>;
template class ArrayStridesBase<3>;
template class ArrayStridesBase<4>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
