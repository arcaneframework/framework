// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationArgs.h                                      (C) 2000-2023 */
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

namespace Arccore
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

  Int8 device() const { return m_device; }
  void setDevice(Int8 device) { m_device = device; }

  ArrayDebugInfo* debugInfo() const { return m_debug_info; }
  void setDebugInfo(ArrayDebugInfo* v) { m_debug_info = v; }

  String arrayName() const;

 private:

  eMemoryLocationHint m_memory_location_hint = eMemoryLocationHint::None;
  Int8 m_device = (-1);
  ArrayDebugInfo* m_debug_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
