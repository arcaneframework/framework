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

template class NumArray<Real, 4>;
template class NumArray<Real, 3>;
template class NumArray<Real, 2>;
template class NumArray<Real, 1>;

template class ArrayStridesBase<1>;
template class ArrayStridesBase<2>;
template class ArrayStridesBase<3>;
template class ArrayStridesBase<4>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
