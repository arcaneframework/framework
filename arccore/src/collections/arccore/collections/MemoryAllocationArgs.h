// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationArgs.h                                      (C) 2000-2025 */
/*                                                                           */
/* Arguments des méthodes de IMemoryAllocator.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_MEMORYALLOCATIONARGS_H
#define ARCCORE_COLLECTIONS_MEMORYALLOCATIONARGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/MemoryAllocationOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe contenant des informations pour spécialiser les allocations.
 */
class ARCCORE_COLLECTIONS_EXPORT MemoryAllocationArgs
{
 public:

  void setMemoryLocationHint(eMemoryLocationHint mem_advice) { m_memory_location_hint = mem_advice; }
  eMemoryLocationHint memoryLocationHint() const { return m_memory_location_hint; }

  void setHostDeviceMemoryLocation(eHostDeviceMemoryLocation v) { m_host_device_memory_location = v; }
  eHostDeviceMemoryLocation hostDeviceMemoryLocation() const { return m_host_device_memory_location; }

  Int16 device() const { return m_device; }
  void setDevice(Int16 device) { m_device = device; }

  ArrayDebugInfo* debugInfo() const { return m_debug_info; }
  void setDebugInfo(ArrayDebugInfo* v) { m_debug_info = v; }

  // RunQueue associée à l'allocation. Peut-être nulle.
  RunQueue* runQueue() const { return m_run_queue; }
  void setRunQueue(RunQueue* v) { m_run_queue = v; }

  String arrayName() const;

 private:

  eMemoryLocationHint m_memory_location_hint = eMemoryLocationHint::None;
  eHostDeviceMemoryLocation m_host_device_memory_location = eHostDeviceMemoryLocation::Unknown;
  Int16 m_device = (-1);
  RunQueue* m_run_queue = nullptr;
  ArrayDebugInfo* m_debug_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
