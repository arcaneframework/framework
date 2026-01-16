// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.cc                                              (C) 2000-2026 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/MemoryUtils.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/common/MemoryAllocationOptions.h"
#include "arccore/common/internal/SpecificMemoryCopyList.h"
#include "arccore/common/internal/MemoryUtilsInternal.h"
#include "arccore/common/internal/MemoryResourceMng.h"

// Pour std::memmove
#include <cstring>

// TODO: ajouter statistiques sur les tailles de 'datatype' utilisées.

/*!
 * \file MemoryUtils.h
 *
 * \brief Fonctions utilitaires de gestion mémoire.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using RunQueue = Accelerator::RunQueue;
using Impl::ISpecificMemoryCopyList;
using Impl::GlobalMemoryCopyList;

/*!
 * \namespace MemoryUtils
 *
 * \brief Fonctions utilitaires de gestion mémoire.
 */

namespace
{
  IMemoryAllocator* global_accelerator_host_memory_allocator = nullptr;
  MemoryResourceMng global_default_data_memory_resource_mng;
  IMemoryRessourceMng* global_data_memory_resource_mng = nullptr;
  eMemoryResource global_data_memory_resource = eMemoryResource::Host;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  ISpecificMemoryCopyList* _getDefaultCopyList(const RunQueue* queue)
  {
    return GlobalMemoryCopyList::getDefault(queue);
  }
  Int32 _checkDataTypeSize(const TraceInfo& trace, Int32 data_size1, Int32 data_size2)
  {
    if (data_size1 != data_size2)
      throw FatalErrorException(trace, String::format("Datatype size are not equal this={0} v={1}", data_size1, data_size2));
    return data_size1;
  }
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
    ARCCORE_FATAL("Invalid name '{0}' for memory resource. Valid names are "
                  "'Device', 'Host', 'HostPinned' or 'UnifieMemory'.",
                  name);
  return v;
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

IMemoryRessourceMng* MemoryUtils::
setDataMemoryResourceMng(IMemoryRessourceMng* mng)
{
  ARCCORE_CHECK_POINTER(mng);
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

ARCCORE_COMMON_EXPORT IMemoryAllocator* MemoryUtils::
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

MemoryAllocationOptions MemoryUtils::
getAllocatorForMostlyReadOnlyData()
{
  return getDefaultDataAllocator(eMemoryLocationHint::HostAndDeviceMostlyRead);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryUtils::
getAllocator(eMemoryResource mem_resource)
{
  return getDataMemoryResourceMng()->getAllocator(mem_resource);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions MemoryUtils::
getAllocationOptions(eMemoryResource mem_resource)
{
  return MemoryAllocationOptions(getAllocator(mem_resource));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryPool* MemoryUtils::
getMemoryPoolOrNull(eMemoryResource mem_resource)
{
  return getDataMemoryResourceMng()->getMemoryPoolOrNull(mem_resource);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copy(MutableMemoryView destination, eMemoryResource destination_mem,
     ConstMemoryView source, eMemoryResource source_mem, const RunQueue* queue)
{
  IMemoryRessourceMng* mrm = getDataMemoryResourceMng();
  mrm->_internal()->copy(source, destination_mem, destination, source_mem, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyHost(MutableMemoryView destination, ConstMemoryView source)
{
  auto b_source = source.bytes();
  auto b_destination = destination.bytes();
  Int64 source_size = b_source.size();
  if (source_size == 0)
    return;
  Int64 destination_size = b_destination.size();
  if (source_size > destination_size)
    ARCCORE_FATAL("Destination is too small source_size={0} destination_size={1}",
                  source_size, destination_size);
  auto* destination_data = b_destination.data();
  auto* source_data = b_source.data();
  ARCCORE_CHECK_POINTER(destination_data);
  ARCCORE_CHECK_POINTER(source_data);
  std::memmove(destination_data, source_data, source_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyHostWithIndexedDestination(MutableMemoryView destination, ConstMemoryView source,
                               Span<const Int32> indexes)
{
  copyWithIndexedDestination(destination, source, indexes.smallView(), nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedDestination(MutableMemoryView destination, ConstMemoryView source,
                           SmallSpan<const Int32> indexes, RunQueue* queue)
{

  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->copyFrom(one_data_size, { indexes, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fillIndexed(MutableMemoryView destination, ConstMemoryView source,
            SmallSpan<const Int32> indexes, const RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->fill(one_data_size, { indexes, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fill(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->fill(one_data_size, { {}, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyHostWithIndexedSource(MutableMemoryView destination, ConstMemoryView source,
                          Span<const Int32> indexes)
{
  copyWithIndexedSource(destination, source, indexes.smallView(), nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedSource(MutableMemoryView destination, ConstMemoryView source,
                      SmallSpan<const Int32> indexes,
                      RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, source.datatypeSize(), destination.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->copyTo(one_data_size, { indexes, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedDestination(MutableMultiMemoryView destination, ConstMemoryView source,
                           SmallSpan<const Int32> indexes, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  _getDefaultCopyList(queue)->copyFrom(one_data_size, { indexes, destination.views(), source.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fillIndexed(MutableMultiMemoryView destination, ConstMemoryView source,
            SmallSpan<const Int32> indexes, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  _getDefaultCopyList(queue)->fill(one_data_size, { indexes, destination.views(), source.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fill(MutableMultiMemoryView destination, ConstMemoryView source, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  _getDefaultCopyList(queue)->fill(one_data_size, { {}, destination.views(), source.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedSource(MutableMemoryView destination, ConstMultiMemoryView source,
                      SmallSpan<const Int32> indexes, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  _getDefaultCopyList(queue)->copyTo(one_data_size, { indexes, source.views(), destination.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT void
arccorePrintSpecificMemoryStats()
{
  if (arccoreIsCheck()) {
    // N'affiche que pour les tests
    //global_copy_list.printStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
