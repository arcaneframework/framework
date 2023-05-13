// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationOptions.h                                    (C) 2000-2023 */
/*                                                                           */
/* Options pour pour configurer les allocations.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_MEMORYALLOCATIONOPTIONS_H
#define ARCCORE_COLLECTIONS_MEMORYALLOCATIONOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

enum class eMemoryLocationHint : int8_t
{
  None = 0,
  MainlyDevice = 1,
  MainlyHost = 2,
  BothMostlyRead = 3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Options pour pour configurer les allocations.
 */
class ARCCORE_COLLECTIONS_EXPORT MemoryAllocationOptions
{
  friend class ArrayMetaData;

 public:

  MemoryAllocationOptions() = default;

  explicit MemoryAllocationOptions(IMemoryAllocator* allocator)
  : m_allocator(allocator)
  {
  }

  explicit MemoryAllocationOptions(IMemoryAllocator* allocator, eMemoryLocationHint advice)
  : m_allocator(allocator)
  , m_memory_advice(advice)
  {
  }

 public:

  IMemoryAllocator* allocator() const { return m_allocator; }
  void setAllocator(IMemoryAllocator* v) { m_allocator = v; }

  eMemoryLocationHint memoryAdvice() const { return m_memory_advice; }
  void setMemoryAdvice(eMemoryLocationHint mem_advice) { m_memory_advice = mem_advice; }

 private:

  IMemoryAllocator* m_allocator = nullptr;
  eMemoryLocationHint m_memory_advice = eMemoryLocationHint::None;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
