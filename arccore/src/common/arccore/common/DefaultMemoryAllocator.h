// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DefaultMemoryAllocator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Default memory allocator.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_DEFAULTMEMORYALLOCATOR_H
#define ARCCORE_COMMON_DEFAULTMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/IMemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory allocator via malloc/realloc/free.
 *
 * TODO: mark methods as 'final'.
 */
class ARCCORE_COMMON_EXPORT DefaultMemoryAllocator
: public IMemoryAllocator
{
  friend class ArrayMetaData;

 private:

  static DefaultMemoryAllocator shared_null_instance;

 public:

  bool hasRealloc(MemoryAllocationArgs) const override;
  AllocatedMemoryInfo allocate(MemoryAllocationArgs, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs, AllocatedMemoryInfo ptr) override;
  Int64 adjustedCapacity(MemoryAllocationArgs, Int64 wanted_capacity, Int64 element_size) const override;
  size_t guaranteedAlignment(MemoryAllocationArgs) const override { return 0; }
  eMemoryResource memoryResource() const override { return eMemoryResource::Host; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory allocator via malloc/realloc/free with listing output.
 *
 * This allocator is primarily used for debugging purposes.
 * Information output is done via std::cout.
 */
class ARCCORE_COMMON_EXPORT PrintableMemoryAllocator
: public DefaultMemoryAllocator
{
  using Base = DefaultMemoryAllocator;

 public:

  using IMemoryAllocator::allocate;
  using IMemoryAllocator::deallocate;
  using IMemoryAllocator::reallocate;

 public:

  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
