// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryAllocator.h                                          (C) 2000-2026 */
/*                                                                           */
/* Memory allocator interface.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_IMEMORYALLOCATOR_H
#define ARCCORE_COMMON_IMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/MemoryAllocationArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a memory allocator.
 *
 * This class defines an interface for the memory allocation used
 * by Arccore array classes (Array, UniqueArray).
 *
 * An instance of this class must remain valid as long as there are
 * arrays using it. Since the allocator is transferred during copies,
 * it is preferable that allocators be static objects whose
 * lifetime is that of the program.
 *
 * Allocators do not have specific mutable state and must function in
 * multi-threading.
 */
class ARCCORE_COMMON_EXPORT IMemoryAllocator
{
 public:

  /*!
   * \brief Destroys the allocator.
   *
   * All objects allocated by the allocator must have been deallocated.
   */
  virtual ~IMemoryAllocator() = default;

 public:

  /*!
   * \brief Indicates whether the allocator supports realloc semantics.
   *
   * Default C allocators (malloc/realloc/free) obviously support
   * realloc, but this is not necessarily the case for specific
   * allocators with memory alignment (such as
   * posix_memalign).
   */
  virtual bool hasRealloc(MemoryAllocationArgs) const { return false; }

  /*!
   * \brief Allocates memory for \a new_size bytes and returns the pointer.
   *
   * The semantics are equivalent to malloc():
   * - \a new_size can be zero, in which case the returned pointer
   * is either null or a specific value
   * - the returned pointer may be null if the memory could not be allocated.
   */
  virtual AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) = 0;

  /*!
   * \brief Reallocates memory for \a new_size bytes and returns the pointer.
   *
   * The pointer \a current_ptr must have been allocated via a call to
   * allocate() or reallocate() on this instance.
   *
   * The semantics of this method are equivalent to realloc():
   * - \a current_ptr may be null, in which case this call is equivalent
   * to allocate().
   * - the returned pointer may be null if the memory could not be allocated.
   */
  virtual AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) = 0;

  /*!
   * \brief Frees the memory whose base address is \a ptr.
   *
   * The pointer \a ptr must have been allocated via a call to
   * allocate() or reallocate() on this instance.
   *
   * The semantics of this method are equivalent to free(), and thus \a ptr
   * may be null, in which case no operation is performed.
   */
  virtual void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) = 0;

  /*!
   * \brief Adjusts the capacity based on the element size.
   *
   * This method is used to optionally modify the number
   * of allocated elements based on their size. This allows, for example,
   * aligned allocators to ensure that the number of elements
   * allocated is a multiple of this alignment.
   */
  virtual Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const = 0;

  /*!
   * \brief Value of the alignment guaranteed by the allocator.
   *
   * This method ensures that an allocator has sufficient alignment
   * for certain operations such as vectorization, for example.
   *
   * If there is no guarantee, it returns 0.
   */
  virtual size_t guaranteedAlignment(MemoryAllocationArgs args) const =0;

  /*!
   * \brief Value of the alignment guaranteed by the allocator.
   *
   * \sa guaranteedAlignment()
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use guaranteedAlignment() instead")
  virtual size_t guarantedAlignment(MemoryAllocationArgs args) const;

  /*!
   * \brief Notifies of a change in instance-specific arguments.
   *
   * \param ptr allocated memory region
   * \param old_args old value of the arguments
   * \param new_args new value of the arguments
   */
  virtual void notifyMemoryArgsChanged(MemoryAllocationArgs old_args, MemoryAllocationArgs new_args, AllocatedMemoryInfo ptr);

  /*!
   * \brief Copies memory between two regions.
   *
   * The default implementation uses std::memcpy().
   *
   * \param args memory region arguments
   * \param destination destination of the copy
   * \param source source of the copy
   */
  virtual void copyMemory(MemoryAllocationArgs args, AllocatedMemoryInfo destination, AllocatedMemoryInfo source);

  //! Memory resource provided by the allocator
  virtual eMemoryResource memoryResource() const { return eMemoryResource::Unknown; }

  /*!
   * \brief Indicates whether calls to the allocator must be performed
   * collectively.
   */
  virtual bool isCollective() const { return false; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
