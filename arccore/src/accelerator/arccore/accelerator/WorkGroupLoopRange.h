// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WorkGroupLoopRange.h                                        (C) 2000-2026 */
/*                                                                           */
/* Loop for hierarchical parallelism.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_WORKGROUPLOOPRANGE_H
#define ARCCORE_ACCELERATOR_WORKGROUPLOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "AcceleratorGlobal.h"
#include "arccore/accelerator/AcceleratorUtils.h"

#if defined(ARCCORE_COMPILING_CUDA)
#include <cooperative_groups.h>
#endif
#if defined(ARCCORE_COMPILING_HIP)
#include <hip/hip_cooperative_groups.h>
#endif

#include <barrier>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

class WorkGroupLoopContextBuilder;
class WorkGroupSequentialForHelper;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Class to manage grid synchronization in multi-thread;
class ThreadGridSynchronizer
{
 private:

  class NullFunc
  {
   public:

    void operator()() const noexcept { /* Nothing to do */ }
  };

 public:

  explicit ThreadGridSynchronizer(Int32 nb_thread)
  : m_barrier(nb_thread)
  {}

 public:

  void sync() { m_barrier.arrive_and_wait(); }

 private:

  std::barrier<NullFunc> m_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_ = Int32>
class HostWorkItem;
template <typename IndexType_ = Int32>
class DeviceWorkItem;
template <typename IndexType_ = Int32>
class SyclDeviceWorkItem;

template <typename IndexType_ = Int32>
class WorkGroupLoopRange;
template <typename IndexType_ = Int32>
class CooperativeWorkGroupLoopRange;

template <typename IndexType_ = Int32>
class WorkGroupLoopContext;
template <typename IndexType_ = Int32>
class CooperativeWorkGroupLoopContext;
template <typename IndexType_ = Int32>
class SyclWorkGroupLoopContext;
template <typename IndexType_ = Int32>
class SyclCooperativeWorkGroupLoopContext;

class HostWorkItemBlock;
class SyclDeviceWorkItemBlock;
class DeviceWorkItemBlock;

class CooperativeHostWorkItemGrid;
class SyclDeviceCooperativeWorkItemGrid;

template <typename Indextype_ = Int32>
class WorkGroupLoopContextBase;
template <typename Indextype_ = Int32>
class SyclWorkGroupLoopContextBase;

template <typename IndexType_ = Int32>
class HostIndexes;
template <typename IndexType_ = Int32>
class DeviceIndexesBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class HostIndexes
{
 public:

  using IndexType = IndexType_;

  class HostWorkItemIterator
  {
   public:

    explicit constexpr HostWorkItemIterator(IndexType loop_index)
    : m_loop_index(loop_index)
    {}
    constexpr IndexType operator*() const { return m_loop_index; }
    HostWorkItemIterator& operator++()
    {
      ++m_loop_index;
      return (*this);
    }
    friend bool operator!=(HostWorkItemIterator a, HostWorkItemIterator b)
    {
      return a.m_loop_index != b.m_loop_index;
    }

   private:

    IndexType m_loop_index = 0;
  };

 public:

  constexpr HostIndexes(IndexType loop_index, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_nb_active_item(nb_active_item)
  {}

 public:

  constexpr HostWorkItemIterator begin() const { return HostWorkItemIterator(m_loop_index); }
  constexpr HostWorkItemIterator end() const { return HostWorkItemIterator(m_loop_index + m_nb_active_item); }

 private:

  IndexType m_loop_index = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class DeviceIndexesBase
{
 public:

  using IndexType = IndexType_;

  class DeviceWorkItemIterator
  {
   public:

    explicit constexpr DeviceWorkItemIterator(IndexType loop_index, Int32 grid_size)
    : m_loop_index(loop_index)
    , m_grid_size(grid_size)
    {}
    constexpr IndexType operator*() const { return m_loop_index; }
    ARCCORE_HOST_DEVICE DeviceWorkItemIterator& operator++()
    {
      m_loop_index += m_grid_size;
      return (*this);
    }
    friend constexpr bool operator!=(DeviceWorkItemIterator a, DeviceWorkItemIterator b)
    {
      return a.m_loop_index != b.m_loop_index;
    }

   private:

    IndexType m_loop_index = 0;
    Int32 m_grid_size = 0;
  };

  class DeviceWorkItemSentinel
  {
   public:

    explicit constexpr DeviceWorkItemSentinel(IndexType total_size)
    : m_total_size(total_size)
    {}
    friend constexpr bool operator!=(DeviceWorkItemIterator a, DeviceWorkItemSentinel b)
    {
      return *a < b.m_total_size;
    }

   private:

    IndexType m_total_size = 0;
  };
};

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

template <typename IndexType_>
class DeviceIndexes
: public DeviceIndexesBase<IndexType_>
{
 public:

  using IndexType = IndexType_;
  using DeviceWorkItemIterator = DeviceIndexesBase<IndexType_>::DeviceWorkItemIterator;
  using DeviceWorkItemSentinel = DeviceIndexesBase<IndexType_>::DeviceWorkItemSentinel;

 public:

  explicit constexpr DeviceIndexes(IndexType total_size)
  : m_total_size(total_size)
  {}

 public:

  __device__ DeviceWorkItemIterator begin() const
  {
    return DeviceWorkItemIterator(blockDim.x * blockIdx.x + threadIdx.x, blockDim.x * gridDim.x);
  }
  constexpr __device__ DeviceWorkItemSentinel end() const
  {
    return DeviceWorkItemSentinel(m_total_size);
  }

 private:

  IndexType m_total_size = 0;
};

#endif

#if defined(ARCCORE_COMPILING_SYCL)

template <typename IndexType_>
class SyclDeviceIndexes
: public DeviceIndexesBase<IndexType_>
{
 public:

  using IndexType = IndexType_;
  using DeviceWorkItemIterator = DeviceIndexesBase<IndexType_>::DeviceWorkItemIterator;
  using DeviceWorkItemSentinel = DeviceIndexesBase<IndexType_>::DeviceWorkItemSentinel;

 public:

  SyclDeviceIndexes(sycl::nd_item<1> nd_item, IndexType total_size)
  : m_nd_item(nd_item)
  , m_total_size(total_size)
  {}

 public:

  DeviceWorkItemIterator begin() const
  {
    IndexType index = static_cast<IndexType>(m_nd_item.get_group(0) * m_nd_item.get_local_range(0) + m_nd_item.get_local_id(0));
    Int32 grid_size = static_cast<Int32>(m_nd_item.get_local_range(0) * m_nd_item.get_group_range(0));
    return DeviceWorkItemIterator(index, grid_size);
  }
  constexpr DeviceWorkItemSentinel end() const { return DeviceWorkItemSentinel(m_total_size); }

 private:

  sycl::nd_item<1> m_nd_item;
  IndexType m_total_size = 0;
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a WorkItem on the host within a WorkGroupLoopRange or
 * CooperativeWorkGroupLoopRange.
 */
template <typename IndexType_>
class HostWorkItem
{
  template <typename T> friend class WorkGroupLoopContextBase;

 public:

  using IndexType = IndexType_;

 private:

  //! Constructor for the host
  constexpr ARCCORE_HOST_DEVICE HostWorkItem(IndexType loop_index, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_nb_active_item(nb_active_item)
  {}

 public:

  //! Rank of the active WorkItem in its WorkGroup.
  constexpr Int32 rankInBlock() const { return 0; }

  //! Indicates if running on a device
  static constexpr bool isDevice() { return false; }

  //! Loop indexes managed by this WorkItem
  constexpr HostIndexes<IndexType> linearIndexes() const
  {
    return HostIndexes<IndexType>(m_loop_index, m_nb_active_item);
  }

 private:

  IndexType m_loop_index = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a block of WorkItems on the host within a WorkGroupLoopRange
 * or a CooperativeWorkGroupLoopRange.
 */
class HostWorkItemBlock
{
  template <typename T> friend class WorkGroupLoopContextBase;

 private:

  //! Constructor for the host
  constexpr ARCCORE_HOST_DEVICE HostWorkItemBlock(Int32 group_index, Int32 group_size)
  : m_group_size(group_size)
  , m_group_index(group_index)
  {}

 public:

  //! Rank of the WorkItem group in the list of WorkGroups.
  constexpr Int32 groupRank() const { return m_group_index; }

  //! Number of WorkItems in a WorkGroup.
  constexpr Int32 groupSize() const { return m_group_size; }

  //! Indicates if running on a device
  static constexpr bool isDevice() { return false; }

  //! Blocks until all \a WorkItems in the group have arrived here.
  void barrier() {}

 private:

  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a WorkItem on a CUDA or HIP device within a
 * WorkGroupLoopRange or a CooperativeWorkGroupLoopRange.
 */
template <typename IndexType_>
class DeviceWorkItem
{
  friend class WorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  /*!
   * \brief Constructor for the device.
   *
   * This constructor does not need specific information as everything is
   * retrieved via cooperative_groups::this_thread_block()
   */
  explicit __device__ DeviceWorkItem(IndexType total_size)
  : m_thread_block(cooperative_groups::this_thread_block())
  , m_total_size(total_size)
  {}

 public:

  //! Rank of the WorkItem in its WorkGroup.
  __device__ Int32 rankInBlock() const { return m_thread_block.thread_index().x; }

  //! Indicates if running on a device
  static constexpr __device__ bool isDevice() { return true; }

  constexpr __device__ DeviceIndexes<IndexType> linearIndexes() const
  {
    return DeviceIndexes<IndexType>(m_total_size);
  }

 private:

  // TODO A supprimer
  cooperative_groups::thread_block m_thread_block;
  IndexType_ m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a WorkItem block in a WorkGroupLoopRange for a CUDA or ROCM device.
 */
class DeviceWorkItemBlock
{
  template <typename T> friend class WorkGroupLoopContextBase;

 private:

  /*!
   * \brief Constructor for the device.
   *
   * This constructor does not need specific information because everything is
   * retrieved via cooperative_groups::this_thread_block()
   */
  explicit __device__ DeviceWorkItemBlock()
  : m_thread_block(cooperative_groups::this_thread_block())
  {}

 public:

  //! Rank of the WorkItem group in the list of WorkGroups.
  __device__ Int32 groupRank() const { return m_thread_block.group_index().x; }

  //! Number of WorkItems in a WorkGroup.
  __device__ Int32 groupSize() { return m_thread_block.group_dim().x; }

  //! Blocks until all \a WorkItems in the group have arrived here.
  __device__ void barrier() { m_thread_block.sync(); }

  //! Indicates if running on an accelerator
  static constexpr __device__ bool isDevice() { return true; }

 private:

  cooperative_groups::thread_block m_thread_block;
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for WorkGroupLoopContext and CooperativeWorkGroupLoopContext.
 */
template <typename IndexType_>
class WorkGroupLoopContextBase
{
 public:

  using IndexType = IndexType_;

 protected:

  //! This constructor is used in the host implementation.
  constexpr WorkGroupLoopContextBase(IndexType loop_index, Int32 group_index, Int32 group_size,
                                     Int32 nb_active_item, IndexType total_size)
  : m_loop_index(loop_index)
  , m_total_size(total_size)
  , m_group_index(group_index)
  , m_group_size(group_size)
  , m_nb_active_item(nb_active_item)
  {
  }

  // This constructor is only used on the device
  // It does nothing because useful values are retrieved via
  // cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE WorkGroupLoopContextBase(IndexType total_size)
  : m_total_size(total_size)
  {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Current group. For CUDA/ROCM, this is a thread block.
  __device__ DeviceWorkItemBlock block() const { return DeviceWorkItemBlock(); }
  //! Active WorkItem. For CUDA/ROCM, this is a thread.
  __device__ DeviceWorkItem<IndexType> workItem() const { return DeviceWorkItem<IndexType>(m_total_size); }
#else
  //! Current group
  HostWorkItemBlock block() const { return HostWorkItemBlock(m_group_index, m_group_size); }
  //! Active WorkItem
  HostWorkItem<IndexType> workItem() const { return { m_loop_index, m_nb_active_item }; }
#endif

 protected:

  IndexType m_loop_index = 0;
  IndexType m_total_size = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution context of a command on a set of blocks.
 *
 * This class is used for the host (sequential and multi-threaded) and
 * for CUDA and ROCM/HIP.
 * The group() method is different on the accelerator and on the host, which
 * allows for specialized command processing.
 */
template <typename IndexType_>
class WorkGroupLoopContext
: public WorkGroupLoopContextBase<IndexType_>
{
  // For accessing constructors
  template <typename T> friend class WorkGroupLoopRange;
  friend Impl::WorkGroupSequentialForHelper;
  friend Impl::WorkGroupLoopContextBuilder;
  using BaseClass = WorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  //! This constructor is used in the host implementation.
  explicit constexpr WorkGroupLoopContext(IndexType loop_index, Int32 group_index, Int32 group_size,
                                          Int32 nb_active_item, IndexType total_size,
                                          [[maybe_unused]] Int32 nb_block,
                                          [[maybe_unused]] Impl::ThreadGridSynchronizer* syncer)
  : BaseClass(loop_index, group_index, group_size, nb_active_item, total_size)
  {
  }

  // This constructor is only used on the device
  // It does nothing because useful values are retrieved via
  // cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE WorkGroupLoopContext(IndexType total_size)
  : BaseClass(total_size)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

/*
 * Implementation for SYCL.
 *
 * The equivalent of \a cooperative_groups::thread_group() with SYCL
 * is \a sycl::nd_item<1>. It is more complicated to use for two
 * reasons:
 *
 * - there is no equivalent of \a cooperative_groups::this_thread_block()
 * in SYCL. You must use the value of \a sycl::nb_item<1> passed as an
 * argument to the compute kernel.
 * - there are no default constructors for \a sycl::nb_item<1>.
 *
 * To circumvent these two problems, a specific type is used to manage
 * kernels in SYCL. Fortunately, it is possible to use lambda templates with
 * SYCL. Therefore, two types are used to manage kernels depending on whether
 * running on the SYCL device or on the host.
 *
 * TODO: check if it is possible with the SYCL_DEVICE_ONLY macro to have the
 * same type containing different fields
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a WorkItem for a Sycl device in a WorkGroupLoopRange or
 * a CooperativeWorkGroupLoopRange.
 */
template <typename IndexType_>
class SyclDeviceWorkItem
{
  friend SyclWorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  explicit SyclDeviceWorkItem(sycl::nd_item<1> nd_item, IndexType total_size)
  : m_nd_item(nd_item)
  , m_total_size(total_size)
  {
  }

 public:

  //! Rank of the active WorkItem in the WorkGroup.
  Int32 rankInBlock() const { return static_cast<Int32>(m_nd_item.get_local_id(0)); }

  //! Indicates if running on an accelerator
  static constexpr bool isDevice() { return true; }

  SyclDeviceIndexes<IndexType> linearIndexes() const
  {
    return SyclDeviceIndexes<IndexType>(m_nd_item, m_total_size);
  }

 private:

  sycl::nd_item<1> m_nd_item;
  IndexType m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a WorkItem block in a WorkGroupLoopRange for a Sycl device.
 */
class SyclDeviceWorkItemBlock
{
  template <typename T> friend class SyclWorkGroupLoopContextBase;

 private:

  explicit SyclDeviceWorkItemBlock(sycl::nd_item<1> nd_item)
  : m_nd_item(nd_item)
  {
  }

 public:

  //! Rank of the WorkItem group in the list of WorkGroups.
  Int32 groupRank() const { return static_cast<Int32>(m_nd_item.get_group(0)); }

  //! Number of WorkItems in a WorkGroup.
  Int32 groupSize() { return static_cast<Int32>(m_nd_item.get_local_range(0)); }

  //! Blocks until all \a WorkItems in the group have arrived here.
  void barrier() { m_nd_item.barrier(); }

  //! Indicates if running on an accelerator
  static constexpr bool isDevice() { return true; }

 private:

  sycl::nd_item<1> m_nd_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution context of a WorkGroupLoopRange for Sycl.
 *
 * This class is used only for the eAcceleratorPolicy::SYCL execution policy.
 */
template <typename IndexType_>
class SyclWorkGroupLoopContextBase
{
  friend WorkGroupLoopRange<IndexType_>;

 public:

  using IndexType = IndexType_;

 protected:

  // This constructor is only used on the device
  explicit SyclWorkGroupLoopContextBase(sycl::nd_item<1> n, IndexType total_size)
  : m_nd_item(n)
  , m_total_size(total_size)
  {
  }

 public:

  //! Current group
  SyclDeviceWorkItemBlock block() const { return SyclDeviceWorkItemBlock(m_nd_item); }

  //! Current WorkItem
  SyclDeviceWorkItem<IndexType_> workItem() const
  {
    return SyclDeviceWorkItem<IndexType_>(m_nd_item, m_total_size);
  }

 protected:

  sycl::nd_item<1> m_nd_item;
  IndexType m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution context of a WorkGroupLoopRange for Sycl.
 *
 * This class is used only for the eAcceleratorPolicy::SYCL execution policy.
 */
template <typename IndexType_>
class SyclWorkGroupLoopContext
: public SyclWorkGroupLoopContextBase<IndexType_>
{
  friend WorkGroupLoopRange<IndexType_>;
  friend Impl::WorkGroupLoopContextBuilder;

 public:

  using IndexType = IndexType_;

 private:

  // This constructor is only used on the device
  explicit SyclWorkGroupLoopContext(sycl::nd_item<1> nd_item, IndexType total_size)
  : SyclWorkGroupLoopContextBase<IndexType_>(nd_item, total_size)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCCORE_COMPILING_SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Iteration range of a loop using hierarchical parallelism.
 *
 * This class is the base class for WorkGroupLoopRange and
 * CooperativeWorkGroupLoopRange.
 *
 * setBlockSize() must be called to set the block size.
 * This can be done by the developer or automatically when launching the
 * command.
 *
 * The iteration range contains nbElement() and is decomposed into \a nbBlock()
 * WorkGroups, each containing \a blockSize() WorkItems.
 *
 * \note On the accelerator, the value of \a blockSize() depends on the
 * accelerator architecture. To be portable, this value must be between
 * 32 and 1024 and be a multiple of 32.
 *
 */
template <bool IsCooperativeLaunch, typename IndexType_>
class WorkGroupLoopRangeBase
{
 public:

  using IndexType = IndexType_;

 public:

  WorkGroupLoopRangeBase() = default;
  explicit WorkGroupLoopRangeBase(IndexType nb_element)
  : m_nb_element(nb_element)
  {
  }

 public:

  static constexpr bool isCooperativeLaunch() { return IsCooperativeLaunch; }

  //! Number of elements to process
  constexpr IndexType nbElement() const { return m_nb_element; }
  //! Block size
  constexpr IndexType blockSize() const { return m_block_size; }
  /*!
   * \brief Number of blocks.
   *
   * Returns 0 if setBlockSize() has not yet been called.
   */
  constexpr Int32 nbBlock() const { return m_nb_block; }

  /*!
   * \brief Sets the block size.
   *
   * \a nb_block must be a multiple of 32.
   */
  ARCCORE_ACCELERATOR_EXPORT void setBlockSize(IndexType nb_block);

  //! Sets the block size based on the \a command
  ARCCORE_ACCELERATOR_EXPORT void setBlockSize(RunCommand& command);

 private:

  IndexType m_nb_element = 0;
  IndexType m_block_size = 0;
  Int32 m_nb_block = 0;

 private:

  void _setNbBlock();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Iteration range of a loop using hierarchical parallelism.
 *
 * \sa WorkGroupLoopRangeBase
 */
template <typename IndexType_>
class WorkGroupLoopRange
: public WorkGroupLoopRangeBase<false, IndexType_>
{
 public:

  using LoopIndexType = WorkGroupLoopContext<IndexType_>;
  using IndexType = IndexType_;

 public:

  WorkGroupLoopRange() = default;
  explicit WorkGroupLoopRange(IndexType total_nb_element)
  : WorkGroupLoopRangeBase<false, IndexType_>(total_nb_element)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
