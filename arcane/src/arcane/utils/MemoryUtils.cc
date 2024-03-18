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
getAllocationOptions(eMemoryRessource mem_ressource)
{
  IMemoryAllocator* allocator = platform::getDataMemoryRessourceMng()->getAllocator(mem_ressource);
  return MemoryAllocationOptions(allocator);
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
copy(MutableMemoryView destination, eMemoryRessource destination_mem,
     ConstMemoryView source, eMemoryRessource source_mem, const RunQueue* queue)
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  mrm->_internal()->copy(source, destination_mem, destination, source_mem, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
