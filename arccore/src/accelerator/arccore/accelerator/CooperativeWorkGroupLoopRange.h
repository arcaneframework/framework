// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CooperativeWorkGroupLoopRange.h                             (C) 2000-2026 */
/*                                                                           */
/* Loop for cooperative hierarchical parallelism.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_COOPERATIVEWORKGROUPLOOPRANGE_H
#define ARCCORE_ACCELERATOR_COOPERATIVEWORKGROUPLOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/WorkGroupLoopRange.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a WorkItem grid in a
 * CooperativeWorkGroupLoopRange for the host.
 *
 * This class only has a barrier() method which performs
 * a barrier on all participating threads in multi-threaded mode.
 */
class CooperativeHostWorkItemGrid
{
  template<typename T> friend class CooperativeWorkGroupLoopContext;

 private:

  //! Constructor for the host
  explicit CooperativeHostWorkItemGrid(Int32 nb_block, Impl::ThreadGridSynchronizer* syncer)
  : m_nb_block(nb_block)
  , m_syncer(syncer)
  {}

 public:

  //! Number of blocks in the grid
  Int32 nbBlock() const { return m_nb_block; }

  //! Blocks until all \a WorkItems in the grid have arrived here.
  void barrier()
  {
    if (m_syncer)
      m_syncer->sync();
  }

 private:

  Int32 m_nb_block = 0;
  Impl::ThreadGridSynchronizer* m_syncer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages the WorkItem grid in a
 * CooperativeWorkGroupLoopRange for a CUDA or HIP device.
 */
class CooperativeDeviceWorkItemGrid
{
  template <typename T> friend class CooperativeWorkGroupLoopContext;

 private:

  /*!
   * \brief Constructor for the device.
   *
   * This constructor does not need specific information because everything is
   * retrieved via cooperative_groups::this_grid()
   */
  __device__ CooperativeDeviceWorkItemGrid()
  : m_grid_group(cooperative_groups::this_grid())
  {}

 public:

  //! Number of blocks in the grid
  __device__ Int32 nbBlock() const { return m_grid_group.group_dim().x; }

  //! Blocks until all \a WorkItems in the grid have arrived here.
  __device__ void barrier() { m_grid_group.sync(); }

 private:

  cooperative_groups::grid_group m_grid_group;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution context for a command on a set of blocks.
 *
 * This class is used for the host (sequential and multi-threaded) and
 * for CUDA and ROCM/HIP. The group() method is different on accelerator and on the host, which
 * allows for specialized command processing.
 */
template <typename IndexType_>
class CooperativeWorkGroupLoopContext
: public WorkGroupLoopContextBase<IndexType_>
{
  // For accessing constructors
  friend class CooperativeWorkGroupLoopRange<IndexType_>;
  friend Impl::WorkGroupSequentialForHelper;
  friend Impl::WorkGroupLoopContextBuilder;
  using BaseClass = WorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  //! This constructor is used in the host implementation.
  constexpr CooperativeWorkGroupLoopContext(IndexType loop_index, Int32 group_index,
                                            Int32 group_size, Int32 nb_active_item,
                                            IndexType total_size, Int32 nb_block, Impl::ThreadGridSynchronizer* syncer)
  : BaseClass(loop_index, group_index, group_size, nb_active_item, total_size)
  , m_nb_block(nb_block)
  , m_syncer(syncer)
  {
  }

  // This constructor is only used on the device
  // It does nothing because useful values are retrieved via
  // cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE CooperativeWorkGroupLoopContext(IndexType total_size)
  : BaseClass(total_size)
  {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Current group. For CUDA/ROCM, this is a thread block.
  __device__ CooperativeDeviceWorkItemGrid grid() const { return CooperativeDeviceWorkItemGrid{}; }
#else
  //! Current group
  CooperativeHostWorkItemGrid grid() const { return CooperativeHostWorkItemGrid(m_nb_block, m_syncer); }
#endif

 private:

  Int32 m_nb_block = 0;
  Impl::ThreadGridSynchronizer* m_syncer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Implementation for SYCL.
 */
#if defined(ARCCORE_COMPILING_SYCL)

/*!
 * \brief Manages the WorkItem grid in a CooperativeWorkGroupLoopRange for a Sycl device.
 */
class SyclCooperativeDeviceWorkItemGrid
{
  template <typename T> friend class SyclCooperativeWorkGroupLoopContext;

 private:

  explicit SyclCooperativeDeviceWorkItemGrid(sycl::nd_item<1> n)
  : m_nd_item(n)
  {
  }

 public:

  //! Number of blocks in the grid
  Int32 nbBlock() const { return static_cast<Int32>(m_nd_item.get_group_range(0)); }

  //! Blocks until all \a CooperativeWorkItems in the grid have arrived here.
  void barrier() { /* Not Yet Implemented */ }

 private:

  sycl::nd_item<1> m_nd_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution context of a CooperativeWorkGroupLoopRange for Sycl.
 *
 * This class is used only for the eAcceleratorPolicy::SYCL execution policy.
 */
template <typename IndexType_>
class SyclCooperativeWorkGroupLoopContext
: public SyclWorkGroupLoopContextBase<IndexType_>
{
  friend CooperativeWorkGroupLoopRange<IndexType_>;
  friend Impl::WorkGroupLoopContextBuilder;

 public:

  using IndexType = IndexType_;

 private:

  // This constructor is only used on the device
  explicit SyclCooperativeWorkGroupLoopContext(sycl::nd_item<1> nd_item, IndexType total_size)
  : SyclWorkGroupLoopContextBase<IndexType_>(nd_item, total_size)
  {
  }

 public:

  //! Current grid
  SyclCooperativeDeviceWorkItemGrid grid() const
  {
    return SyclCooperativeDeviceWorkItemGrid(this->m_nd_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCCORE_COMPILING_SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Iteration range of a loop using cooperative hierarchical parallelism.
 *
 * \sa WorkGroupLoopRangeBase
 */
template <typename IndexType_>
class CooperativeWorkGroupLoopRange
: public WorkGroupLoopRangeBase<true, IndexType_>
{
 public:

  using LoopIndexType = CooperativeWorkGroupLoopContext<IndexType_>;
  using IndexType = IndexType_;

 public:

  CooperativeWorkGroupLoopRange() = default;
  explicit CooperativeWorkGroupLoopRange(IndexType total_nb_element)
  : WorkGroupLoopRangeBase<true, IndexType_>(total_nb_element)
  {}

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
