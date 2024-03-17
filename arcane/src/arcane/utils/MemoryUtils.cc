// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryUtils.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryUtils::
getDefaultDataAllocator()
{
  return platform::getDefaultDataAllocator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions MemoryUtils::
getDefaultDataAllocator(eMemoryLocationHint hint)
{
  return MemoryAllocationOptions(getDefaultDataAllocator(), hint);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions MemoryUtils::
getAllocatorForMostlyReadOnlyData()
{
  return getDefaultDataAllocator(eMemoryLocationHint::HostAndDeviceMostlyRead);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MemoryUtils::impl::
computeCapacity(Int64 size)
{
  Int64 new_capacity = size * 2;
  if (size > 5000000)
    new_capacity = static_cast<Int64>(static_cast<double>(size) * 1.2);
  else if (size > 500000)
    new_capacity = static_cast<Int64>(static_cast<double>(size) * 1.5);
  return new_capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue)
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  eMemoryRessource mem_type = eMemoryRessource::Unknown;
  mrm->_internal()->copy(source, mem_type, destination, mem_type, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
