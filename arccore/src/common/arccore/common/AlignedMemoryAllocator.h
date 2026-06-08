// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlignedMemoryAllocator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Default memory allocator.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ALIGNEDMEMORYALLOCATOR_H
#define ARCCORE_COMMON_ALIGNEDMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/IMemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Impl
{
  extern "C++" ARCCORE_COMMON_EXPORT size_t
  adjustMemoryCapacity(size_t wanted_capacity, size_t element_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory allocator with specific memory alignment.
 *
 * This class is used via the two public methods Simd()
 * and CacheLine(), which return an allocator with adequate alignment to
 * allow vectorization, and an allocator aligned to a cache line, respectively.
 */
class ARCCORE_COMMON_EXPORT AlignedMemoryAllocator
: public IMemoryAllocator
{
 private:

  static AlignedMemoryAllocator SimdAllocator;
  static AlignedMemoryAllocator CacheLineAllocator;

 public:

  // TODO: try to find the correct values based on the target.
  // 64 is OK for all x64 architectures for both SIMD and cache line.

  // IMPORTANT: If we change the value here, we must change the alignment
  // size of ArrayImplBase.

  // TODO Currently, only 64 alignment is allowed. To allow other values,
  // the implementation in ArrayImplBase must be modified.

  // TODO mark the methods as 'final'.

  //! Alignment for structures using vectorization
  static constexpr Integer simdAlignment() { return 64; }
  //! Alignment for a cache line.
  static constexpr Integer cacheLineAlignment() { return 64; }

  /*!
   * \brief Allocator guaranteeing alignment to use vectorization on the
   * target platform.
   *
   * This is the alignment for the more restrictive type, and therefore
   * this allocator can be used for all vector structures.
   */
  static AlignedMemoryAllocator* Simd()
  {
    return &SimdAllocator;
  }

  /*!
   * \brief Allocator guaranteeing alignment to a cache line.
   */
  static AlignedMemoryAllocator* CacheLine()
  {
    return &CacheLineAllocator;
  }

 protected:

  explicit AlignedMemoryAllocator(Int32 alignment)
  : m_alignment(static_cast<size_t>(alignment))
  {}

 public:

  bool hasRealloc(MemoryAllocationArgs) const override { return false; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override;
  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const override;
  size_t guaranteedAlignment(MemoryAllocationArgs) const override { return m_alignment; }
  eMemoryResource memoryResource() const override { return eMemoryResource::Host; }

 private:

  size_t m_alignment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
