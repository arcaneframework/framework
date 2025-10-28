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

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/String.h"
#include "arccore/common/internal/IMemoryResourceMngInternal.h"
#include "arcane/utils/internal/MemoryUtilsInternal.h"
#include "arcane/utils/internal/MemoryResourceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MemoryUtils.h
 * \brief Fonctions de gestion mémoire et des allocateurs.
 */
/*!
 * \namespace Arcane::MemoryUtils
 * \brief Espace de noms pour les fonctions de gestion mémoire et des allocateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  IMemoryAllocator* global_accelerator_host_memory_allocator = nullptr;
  MemoryResourceMng global_default_data_memory_resource_mng;
  IMemoryRessourceMng* global_data_memory_resource_mng = nullptr;
  eMemoryResource global_data_memory_resource = eMemoryResource::Host;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eMemoryResource MemoryUtils::
getDefaultDataMemoryResource()
{
  return global_data_memory_resource;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
setDefaultDataMemoryResource(eMemoryResource v)
{
  global_data_memory_resource = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eMemoryResource MemoryUtils::
getMemoryResourceFromName(const String& name)
{
  eMemoryResource v = eMemoryResource::Unknown;
  if (name.null())
    return v;
  if (name == "Device")
    v = eMemoryResource::Device;
  else if (name == "Host")
    v = eMemoryResource::Host;
  else if (name == "HostPinned")
    v = eMemoryResource::HostPinned;
  else if (name == "UnifiedMemory")
    v = eMemoryResource::UnifiedMemory;
  else
    ARCANE_FATAL("Invalid name '{0}' for memory resource. Valid names are "
                 "'Device', 'Host', 'HostPinned' or 'UnifieMemory'.",
                 name);
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryRessourceMng* MemoryUtils::
setDataMemoryResourceMng(IMemoryRessourceMng* mng)
{
  ARCANE_CHECK_POINTER(mng);
  IMemoryRessourceMng* old = global_data_memory_resource_mng;
  global_data_memory_resource_mng = mng;
  return old;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryRessourceMng* MemoryUtils::
getDataMemoryResourceMng()
{
  IMemoryRessourceMng* a = global_data_memory_resource_mng;
  if (!a)
    return &global_default_data_memory_resource_mng;
  return a;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryUtils::
getDefaultDataAllocator()
{
  return getDataMemoryResourceMng()->getAllocator(getDefaultDataMemoryResource());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryUtils::
getDeviceOrHostAllocator()
{
  IMemoryRessourceMng* mrm = getDataMemoryResourceMng();
  IMemoryAllocator* a = mrm->getAllocator(eMemoryResource::Device, false);
  if (a)
    return a;
  return mrm->getAllocator(eMemoryResource::Host);
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

IMemoryAllocator* MemoryUtils::
getAllocator(eMemoryRessource mem_resource)
{
  return getDataMemoryResourceMng()->getAllocator(mem_resource);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions MemoryUtils::
getAllocationOptions(eMemoryRessource mem_resource)
{
  return MemoryAllocationOptions(getAllocator(mem_resource));
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

IMemoryAllocator* MemoryUtils::
getAcceleratorHostMemoryAllocator()
{
  return global_accelerator_host_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryUtils::
setAcceleratorHostMemoryAllocator(IMemoryAllocator* a)
{
  IMemoryAllocator* old = global_accelerator_host_memory_allocator;
  global_accelerator_host_memory_allocator = a;
  return old;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MemoryUtils::impl::
computeCapacity(Int64 size)
{
  double d_size = static_cast<double>(size);
  double d_new_capacity = d_size * 1.8;
  if (size > 5000000)
    d_new_capacity = d_size * 1.2;
  else if (size > 500000)
    d_new_capacity = d_size * 1.5;
  return static_cast<Int64>(d_new_capacity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copy(MutableMemoryView destination, eMemoryRessource destination_mem,
     ConstMemoryView source, eMemoryRessource source_mem, const RunQueue* queue)
{
  IMemoryRessourceMng* mrm = getDataMemoryResourceMng();
  mrm->_internal()->copy(source, destination_mem, destination, source_mem, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
