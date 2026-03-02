// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationOptions.h                                   (C) 2000-2025 */
/*                                                                           */
/* Options pour configurer les allocations.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_MEMORYALLOCATIONOPTIONS_H
#define ARCCORE_COMMON_MEMORYALLOCATIONOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Options pour configurer les allocations.
 */
class ARCCORE_COMMON_EXPORT MemoryAllocationOptions
{
  friend class ArrayMetaData;

 public:

  MemoryAllocationOptions() = default;

  explicit MemoryAllocationOptions(IMemoryAllocator* allocator)
  : m_allocator(allocator)
  {
  }

  MemoryAllocationOptions(IMemoryAllocator* allocator, eMemoryLocationHint mem_hint)
  : m_allocator(allocator)
  , m_memory_location_hint(mem_hint)
  {
  }

  MemoryAllocationOptions(IMemoryAllocator* allocator, eMemoryLocationHint mem_hint, Int8 device)
  : m_allocator(allocator)
  , m_device(device)
  , m_memory_location_hint(mem_hint)
  {
  }

 public:

  MemoryAllocationOptions(const MemoryAllocationOptions& rhs)
  : m_allocator(rhs.m_allocator)
  , m_debug_info(rhs.m_debug_info)
  , m_device(rhs.m_device)
  , m_memory_location_hint(rhs.m_memory_location_hint)
  , m_host_device_memory_location(rhs.m_host_device_memory_location)
  {
    if (m_debug_info)
      _addDebugReference();
  }

  ~MemoryAllocationOptions()
  {
    if (m_debug_info)
      _removeDebugReference();
  }
  MemoryAllocationOptions& operator=(const MemoryAllocationOptions& rhs)
  {
    if (&rhs == this)
      return (*this);
    if (m_debug_info)
      _removeDebugReference();
    m_allocator = rhs.m_allocator;
    m_memory_location_hint = rhs.m_memory_location_hint;
    m_host_device_memory_location = rhs.m_host_device_memory_location;
    m_device = rhs.m_device;
    m_debug_info = rhs.m_debug_info;
    if (m_debug_info)
      _addDebugReference();
    return (*this);
  }

 public:

  IMemoryAllocator* allocator() const { return m_allocator; }
  void setAllocator(IMemoryAllocator* v) { m_allocator = v; }

  eMemoryLocationHint memoryLocationHint() const { return m_memory_location_hint; }
  void setMemoryLocationHint(eMemoryLocationHint mem_advice) { m_memory_location_hint = mem_advice; }

  void setHostDeviceMemoryLocation(eHostDeviceMemoryLocation v) { m_host_device_memory_location = v; }
  eHostDeviceMemoryLocation hostDeviceMemoryLocation() const { return m_host_device_memory_location; }

  Int16 device() const { return m_device; }
  void setDevice(Int16 device) { m_device = device; }

  void setArrayName(const String& name);
  String arrayName() const;

  RunQueue* runQueue() const { return m_queue; }

  //! Arguments pour 'IMemoryAllocator' associés à ces options et à la file \a queue
  MemoryAllocationArgs allocationArgs(RunQueue* queue = nullptr) const;

 public:

  // TODO: A supprimer car ne sert que pour les tests
  friend bool operator==(const MemoryAllocationOptions& a, const MemoryAllocationOptions& b)
  {
    if (a.m_allocator != b.m_allocator)
      return false;
    if (a.m_memory_location_hint != b.m_memory_location_hint)
      return false;
    if (a.m_host_device_memory_location != b.m_host_device_memory_location)
      return false;
    if (a.m_device != b.m_device)
      return false;
    if (a.m_queue != b.m_queue)
      return false;
    return true;
  }

 private:

  IMemoryAllocator* m_allocator = nullptr;
  ArrayDebugInfo* m_debug_info = nullptr;
  Int16 m_device = -1;
  eMemoryLocationHint m_memory_location_hint = eMemoryLocationHint::None;
  eHostDeviceMemoryLocation m_host_device_memory_location = eHostDeviceMemoryLocation::Unknown;
  RunQueue* m_queue = nullptr;

 private:

  void _addDebugReference();
  void _removeDebugReference();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
