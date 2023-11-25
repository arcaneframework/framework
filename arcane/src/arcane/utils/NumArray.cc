// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.cc                                                 (C) 2000-2023 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques sur accélérateur.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
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
  ARCANE_FATAL("Invalid access from '{0}' ressource memory to host memory",(int)r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayBaseCommon::
_memoryAwareCopy(Span<const std::byte> from, eMemoryRessource from_mem,
                 Span<std::byte> to, eMemoryRessource to_mem, RunQueue* queue)
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  mrm->_internal()->copy(ConstMemoryView(from), from_mem, MutableMemoryView(to), to_mem, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayBaseCommon::
_memoryAwareFill(Span<std::byte> to, Int64 nb_element, const void* fill_address,
                 Int32 datatype_size, SmallSpan<const Int32> indexes, RunQueue* queue)
{
  ConstMemoryView fill_value_view(makeConstMemoryView(fill_address, datatype_size, 1));
  MutableMemoryView destination(makeMutableMemoryView(to.data(), datatype_size, nb_element));
  destination.fillIndexes(fill_value_view, indexes, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayBaseCommon::
_memoryAwareFill(Span<std::byte> to, Int64 nb_element, const void* fill_address,
                 Int32 datatype_size, RunQueue* queue)
{
  ConstMemoryView fill_value_view(makeConstMemoryView(fill_address, datatype_size, 1));
  MutableMemoryView destination(makeMutableMemoryView(to.data(), datatype_size, nb_element));
  destination.fill(fill_value_view, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class NumArray<Real, MDDim4>;
template class NumArray<Real, MDDim3>;
template class NumArray<Real, MDDim2>;
template class NumArray<Real, MDDim1>;

template class ArrayStridesBase<1>;
template class ArrayStridesBase<2>;
template class ArrayStridesBase<3>;
template class ArrayStridesBase<4>;

namespace impl
{
  template class NumArrayContainer<Real>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
