// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationOptions.h                                   (C) 2000-2023 */
/*                                                                           */
/* Options pour configurer les allocations.                                  */
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

  explicit MemoryAllocationOptions(IMemoryAllocator* allocator, eMemoryLocationHint mem_hint)
  : m_allocator(allocator)
  , m_memory_location_hint(mem_hint)
  {
  }

 public:

  IMemoryAllocator* allocator() const { return m_allocator; }
  void setAllocator(IMemoryAllocator* v) { m_allocator = v; }

  eMemoryLocationHint memoryLocationHint() const { return m_memory_location_hint; }
  void setMemoryLocationHint(eMemoryLocationHint mem_advice) { m_memory_location_hint = mem_advice; }

  MemoryAllocationArgs allocationArgs() const;

  public:

  friend bool operator==(const MemoryAllocationOptions& a, const MemoryAllocationOptions& b)
  {
    if (a.m_allocator == b.m_allocator)
      return true;
    return (a.m_memory_location_hint == b.m_memory_location_hint);
  }

 private:

  IMemoryAllocator* m_allocator = nullptr;
  eMemoryLocationHint m_memory_location_hint = eMemoryLocationHint::None;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
