// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayMetaData.h                                             (C) 2000-2026 */
/*                                                                           */
/* 1D Array.                                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARRAYMETADATA_H
#define ARCCORE_COMMON_ARRAYMETADATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/MemoryAllocationOptions.h"
#include "arccore/common/MemoryAllocationArgs.h"
#include "arccore/common/IMemoryAllocator.h"
#include "arccore/common/AllocatedMemoryInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 *
 * \brief Array Metadata.
 *
 * This class is used to hold common metadata for all
 * implementations that derive from AbstractArray.
 *
 * Only classes that implement a UniqueArray semantics
 * are allowed to use an allocator other than the default allocator.
 */
class ARCCORE_COMMON_EXPORT ArrayMetaData
{
  // NOTE: The fields of this class are used for the TTF display of totalview.
  // If their order is modified, the copy of this class
  // in Arcane's totalview displayer must be updated.

  template <typename> friend class AbstractArray;
  template <typename> friend class Array2;
  template <typename> friend class Array;
  template <typename> friend class SharedArray;
  template <typename> friend class SharedArray2;
  friend class AbstractArrayBase;
  static IMemoryAllocator* _defaultAllocator();

 public:

  ArrayMetaData()
  : allocation_options(_defaultAllocator())
  {}

 protected:

  //! Number of elements in the array (for 1D arrays)
  Int64 size = 0;
  //! Size of the first dimension (for 2D arrays)
  Int64 dim1_size = 0;
  //! Size of the second dimension (for 2D arrays)
  Int64 dim2_size = 0;
  //! Number of allocated elements
  Int64 capacity = 0;
  //! Memory allocator and associated options
  MemoryAllocationOptions allocation_options;
  //! Number of references on the instance
  Int32 nb_ref = 0;
  //! Information about the physical location of the memory (if known)
  eHostDeviceMemoryLocation m_host_device_memory_location = eHostDeviceMemoryLocation::Unknown;
  //! Indicates if this instance was allocated by the new operator.
  bool is_allocated_by_new = false;
  //! Indicates if this instance is not the null instance (shared by all SharedArray)
  bool is_not_null = false;
  //! Indicates if calls to the allocator must be performed collectively.
  bool is_collective_allocator = false;

 protected:

  IMemoryAllocator* _allocator() const { return allocation_options.m_allocator; }

 public:

  static void throwInvalidMetaDataForSharedArray ARCCORE_NORETURN();
  static void throwNullExpected ARCCORE_NORETURN();
  static void throwNotNullExpected ARCCORE_NORETURN();
  static void throwUnsupportedSpecificAllocator ARCCORE_NORETURN();
  static void overlapError ARCCORE_NORETURN(const void* begin1, Int64 size1,
                                            const void* begin2, Int64 size2);

 protected:

  using MemoryPointer = void*;
  using ConstMemoryPointer = const void*;

 protected:

  MemoryPointer _allocate(Int64 nb, Int64 sizeof_true_type, RunQueue* queue);
  MemoryPointer _reallocate(const AllocatedMemoryInfo& mem_info, Int64 new_capacity, Int64 sizeof_true_type, RunQueue* queue);
  void _deallocate(const AllocatedMemoryInfo& mem_info, RunQueue* queue) noexcept
  {
    if (_allocator()) {
      MemoryAllocationArgs alloc_args = _getAllocationArgs(queue);
      _allocator()->deallocate(alloc_args, mem_info);
    }
  }
  MemoryPointer _changeAllocator(const MemoryAllocationOptions& new_allocator_opt, const AllocatedMemoryInfo& current_info, Int64 sizeof_true_type, RunQueue* queue);
  void _setMemoryLocationHint(eMemoryLocationHint new_hint, void* ptr, Int64 sizeof_true_type);
  void _setHostDeviceMemoryLocation(eHostDeviceMemoryLocation location);
  void _copyFromMemory(MemoryPointer destination, ConstMemoryPointer source, Int64 sizeof_true_type, RunQueue* queue);

 private:

  void _checkAllocator() const;
  MemoryAllocationArgs _getAllocationArgs() const { return allocation_options.allocationArgs(); }
  MemoryAllocationArgs _getAllocationArgs(RunQueue* queue) const
  {
    return allocation_options.allocationArgs(queue);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 *
 * \brief This type is no longer used.
 */
class ArrayImplBase
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 *
 * \brief This class is no longer used.
 */
template <typename T>
class ArrayImplT
: public ArrayImplBase
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
