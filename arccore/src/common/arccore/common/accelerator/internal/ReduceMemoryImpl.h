// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReduceMemoryImpl.h                                          (C) 2000-2025 */
/*                                                                           */
/* Memory management for reductions.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_REDUCEMEMORYIMPL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_REDUCEMEMORYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/IReduceMemoryImpl.h"

#include "arccore/common/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ReduceMemoryImpl
: public IReduceMemoryImpl
{
 public:

  explicit ReduceMemoryImpl(RunCommandImpl* p);

 public:

  void allocateReduceDataMemory(Int32 data_type_size) override;
  void setGridSizeAndAllocate(Int32 grid_size) override
  {
    m_grid_size = grid_size;
    _allocateGridDataMemory();
  }
  Int32 gridSize() const override { return m_grid_size; }

  GridMemoryInfo gridMemoryInfo() override
  {
    return m_grid_memory_info;
  }
  void release() override;

 private:

  RunCommandImpl* m_command = nullptr;

  //! Allocation for the reduced data in host memory
  UniqueArray<std::byte> m_host_memory_bytes;

  //! Size allocated for \a m_device_memory
  Int64 m_size = 0;

  //! Current grid size (number of blocks)
  Int32 m_grid_size = 0;

  //! Current data size
  Int64 m_data_type_size = 0;

  GridMemoryInfo m_grid_memory_info;

  //! Array containing the reduction value for each block of a grid
  UniqueArray<Byte> m_grid_buffer;

  //! Buffer to store the identity value
  UniqueArray<std::byte> m_identity_buffer;

  /*!
   * \brief Array of 1 unsigned integer containing the number of grids that have already
   * performed the reduction.
   */
  UniqueArray<unsigned int> m_grid_device_count;

 private:

  void _allocateGridDataMemory();
  void _allocateMemoryForGridDeviceCount();
  void _setReducePolicy();
  void _allocateMemoryForReduceData(Int32 new_size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
