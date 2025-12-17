// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DefaultMemoryAllocator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Allocateur mémoire par défaut.                                            */
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
 * \brief Allocateur mémoire via malloc/realloc/free.
 *
 * TODO: marquer les méthodes comme 'final'.
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
 * \brief Allocateur mémoire via malloc/realloc/free avec impression listing.
 *
 * Cet allocateur est principalement utilisé à des fins de debugging.
 * La sortie des informations se fait sur std::cout.
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

