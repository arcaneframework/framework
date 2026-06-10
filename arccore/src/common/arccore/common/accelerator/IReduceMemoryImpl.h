// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IReduceMemoryImpl.h                                         (C) 2000-2026 */
/*                                                                           */
/* Interface for memory management for reductions.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_IREDUCEMEMORYIMPL_H
#define ARCCORE_COMMON_ACCELERATOR_IREDUCEMEMORYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for memory management for reductions.
 */
class ARCCORE_COMMON_EXPORT IReduceMemoryImpl
{
 public:

  //! Memory information for reduction on accelerators
  struct GridMemoryInfo
  {
    //! Memory allocated for reduction on a grid (of size nb_block * sizeof(T))
    MutableMemoryView m_grid_memory_values;
    //! Integer used to count the number of blocks that have already
    //! completed their part of the reduction
    unsigned int* m_grid_device_count = nullptr;
    /*!
     * \brief Pointer to the host memory containing the reduced value.
     *
     * This memory is pinned and is therefore accessible from the accelerator.
     */
    void* m_host_memory_for_reduced_value = nullptr;
  };

 public:

  virtual ~IReduceMemoryImpl() = default;

 public:

  /*!
   * \brief Allocates memory for a data item that needs to be reduced.
   *
   * \a data_type_size is the size of the data.
   */
  virtual void allocateReduceDataMemory(Int32 data_type_size) = 0;

  //! Sets the GPU grid size (the number of blocks) and allocates memory
  virtual void setGridSizeAndAllocate(Int32 grid_size) = 0;

  //! GPU grid size (number of blocks)
  virtual Int32 gridSize() const = 0;

  //! Information about the memory used by the reduction
  virtual GridMemoryInfo gridMemoryInfo() = 0;

  //! Releases the instance.
  virtual void release() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
