// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchImpl.h                                      (C) 2000-2026 */
/*                                                                           */
/* Implementation of a RunCommand for hierarchical parallelism.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCHIMPL_H
#define ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCHIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "AcceleratorGlobal.h"
#include "arccore/common/SequentialFor.h"
#include "arccore/common/StridedLoopRanges.h"
#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/concurrency/ParallelFor.h"

#include "arccore/accelerator/WorkGroupLoopRange.h"
#include "arccore/accelerator/CooperativeWorkGroupLoopRange.h"
#include "arccore/accelerator/KernelLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information of a loop using hierarchical parallelism
 * on the host.
 */
template <typename IndexType_>
class HostLaunchLoopRangeBase
{
 public:

  using IndexType = IndexType_;

 public:

  ARCCORE_ACCELERATOR_EXPORT
  HostLaunchLoopRangeBase(IndexType total_size, Int32 nb_group, IndexType block_size);

 public:

  //! Number of elements to process
  constexpr IndexType nbElement() const { return m_total_size; }
  //! Block size
  constexpr IndexType blockSize() const { return m_block_size; }
  //! Number of blocks
  constexpr Int32 nbBlock() const { return m_nb_block; }
  //! Number of elements in the last block
  constexpr IndexType lastBlockSize() const { return m_last_block_size; }
  //! Number of active items for the i-th block
  constexpr IndexType nbActiveItem(Int32 i) const
  {
    return ((i + 1) != m_nb_block) ? m_block_size : m_last_block_size;
  }
  //! Grid synchronizer (non-null only in cooperative multi-threading)
  ThreadGridSynchronizer* threadGridSynchronizer() const
  {
    return m_thread_grid_synchronizer;
  }
  void setThreadGridSynchronizer(ThreadGridSynchronizer* v)
  {
    m_thread_grid_synchronizer = v;
  }

 private:

  //! This instance is managed by arcaneParallelFor(HostLaunchLoopRange<>...)
  ThreadGridSynchronizer* m_thread_grid_synchronizer = nullptr;
  IndexType m_total_size = 0;
  IndexType m_block_size = 0;
  IndexType m_last_block_size = 0;
  Int32 m_nb_block = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename WorkGroupLoopRangeType_>
class HostLaunchLoopRange
: public HostLaunchLoopRangeBase<typename WorkGroupLoopRangeType_::IndexType>
{
 public:

  using WorkGroupLoopRangeType = WorkGroupLoopRangeType_;
  using IndexType = typename WorkGroupLoopRangeType_::IndexType;
  using BaseClass = HostLaunchLoopRangeBase<typename WorkGroupLoopRangeType_::IndexType>;

 public:

  explicit HostLaunchLoopRange(const WorkGroupLoopRangeType& bounds)
  : BaseClass(bounds.nbElement(), bounds.nbBlock(), bounds.blockSize())
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class WorkGroupLoopContextBuilder
{
 public:

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

  template <typename IndexType_> static constexpr ARCCORE_HOST_DEVICE WorkGroupLoopContext<IndexType_>
  build(const WorkGroupLoopRange<IndexType_>& loop_range)
  {
    return WorkGroupLoopContext<IndexType_>(loop_range.nbElement());
  }

  template <typename IndexType_> static constexpr ARCCORE_HOST_DEVICE CooperativeWorkGroupLoopContext<IndexType_>
  build(const CooperativeWorkGroupLoopRange<IndexType_>& loop_range)
  {
    return CooperativeWorkGroupLoopContext<IndexType_>(loop_range.nbElement());
  }

#endif

#if defined(ARCCORE_COMPILING_SYCL)

  template <typename IndexType_> static SyclWorkGroupLoopContext<IndexType_>
  build(const WorkGroupLoopRange<IndexType_>& loop_range, sycl::nd_item<1> id)
  {
    return SyclWorkGroupLoopContext<IndexType_>(id, loop_range.nbElement());
  }

  template <typename IndexType_> static SyclCooperativeWorkGroupLoopContext<IndexType_>
  build(const CooperativeWorkGroupLoopRange<IndexType_>& loop_range, sycl::nd_item<1> id)
  {
    return SyclCooperativeWorkGroupLoopContext<IndexType_>(id, loop_range.nbElement());
  }
#endif
};

#if defined(ARCCORE_COMPILING_SYCL)

// To indicate that sycl::nd_item must always be used (and never sycl::id)
// as an argument with 'WorkGroupLoopRange.
template <typename IndexType_>
class IsAlwaysUseSyclNdItem<StridedLoopRanges<WorkGroupLoopRange<IndexType_>>>
: public std::true_type
{
};
// To indicate that sycl::nd_item must always be used (and never sycl::id)
// as an argument with 'CooperativeWorkGroupLoopRange.
template <typename IndexType_>
class IsAlwaysUseSyclNdItem<StridedLoopRanges<CooperativeWorkGroupLoopRange<IndexType_>>>
: public std::true_type
{
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class to execute a portion of the loop sequentially on the host.
 */
class WorkGroupSequentialForHelper
{
 public:

  //! Applies the functor \a func on a sequential loop.
  template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> static void
  apply(Int32 begin_index, Int32 nb_loop, HostLaunchLoopRange<LoopBoundType> bounds,
        const Lambda& func, RemainingArgs... remaining_args)
  {
    using LoopIndexType = LoopBoundType::LoopIndexType;
    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtBegin(remaining_args...);
    const Int32 group_size = bounds.blockSize();
    Int32 loop_index = begin_index * group_size;
    for (Int32 i = begin_index; i < (begin_index + nb_loop); ++i) {
      // For the last loop iteration, the number of active elements may be
      // less than the group size if \a total_nb_element is not
      // a multiple of \a group_size.
      Int32 nb_active = bounds.nbActiveItem(i);
      LoopIndexType li(loop_index, i, group_size, nb_active, bounds.nbElement(), bounds.nbBlock(), bounds.threadGridSynchronizer());
      func(li, remaining_args...);
      loop_index += group_size;
    }

    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtEnd(remaining_args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

// We use 'Argument dependent lookup' to find 'arcaneGetLoopIndexCudaHip'
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> __global__ static void
doHierarchicalLaunchCudaHip(LoopBoundType bounds, Lambda func, RemainingArgs... remaining_args)
{
  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;

  CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);
  // TODO: check if this test is necessary
  if (i < bounds.nbOriginalElement()) {
    func(WorkGroupLoopContextBuilder::build(bounds.originalLoop()), remaining_args...);
  }
  CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
};

#endif

#if defined(ARCCORE_COMPILING_SYCL)

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs>
class doHierarchicalLaunchSycl
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<std::byte> shared_memory,
                  LoopBoundType bounds, Lambda func,
                  RemainingArgs... remaining_args) const
  {
    Int32 i = static_cast<Int32>(x.get_global_id(0));
    SyclKernelRemainingArgsHelper::applyAtBegin(x, shared_memory, remaining_args...);
    // TODO: check if this test is necessary
    if (i < bounds.nbOriginalElement()) {
      func(WorkGroupLoopContextBuilder::build(bounds.originalLoop(), x), remaining_args...);
    }
    SyclKernelRemainingArgsHelper::applyAtEnd(x, shared_memory, remaining_args...);
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda \a func on a loop \a bounds.
 *
 * The lambda \a func is applied to the \a command.
 * \a bound is the loop type. Supported types are:
 *
 * - WorkGroupLoopRange
 * - CooperativeWorkGroupLoopRange
 *
 * Additional arguments \a other_args are used to support
 * features such as reductions (ReducerSum2, ReducerMax2, ...)
 * or local memory management (via LocalMemory).
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
_doHierarchicalLaunch(RunCommand& command, LoopBoundType bounds,
                      const Lambda& func, const RemainingArgs&... other_args)
{
  Int64 nb_orig_element = bounds.nbElement();
  if (nb_orig_element == 0)
    return;
  const eExecutionPolicy exec_policy = command.executionPolicy();
  // In cooperative mode, setBlockSize() must always be called
  // to ensure that the block size is consistent on the host
  // (in sequential mode, only one block is needed in this case).
  if ((bounds.blockSize() == 0) || bounds.isCooperativeLaunch())
    bounds.setBlockSize(command);
  using TrueLoopBoundType = StridedLoopRanges<LoopBoundType>;
  TrueLoopBoundType bounds2(bounds);
  if (isAcceleratorPolicy(exec_policy)) {
    command.addNbThreadPerBlock(bounds.blockSize());
    bounds2.setNbStride(command.nbStride());
  }

  using HostLoopBoundType = HostLaunchLoopRange<LoopBoundType>;

  Impl::RunCommandLaunchInfo launch_info(command, bounds2.strideValue(), bounds.isCooperativeLaunch());
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    ARCCORE_KERNEL_CUDA_FUNC((Impl::doHierarchicalLaunchCudaHip<TrueLoopBoundType, Lambda, RemainingArgs...>),
                             launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::HIP:
    ARCCORE_KERNEL_HIP_FUNC((Impl::doHierarchicalLaunchCudaHip<TrueLoopBoundType, Lambda, RemainingArgs...>),
                            launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::SYCL:
    ARCCORE_KERNEL_SYCL_FUNC((Impl::doHierarchicalLaunchSycl<TrueLoopBoundType, Lambda, RemainingArgs...>{}),
                             launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::Sequential: {
    HostLoopBoundType host_bounds(bounds);
    arccoreSequentialFor(host_bounds, func, other_args...);
  } break;
  case eExecutionPolicy::Thread: {
    HostLoopBoundType host_bounds(bounds);
    arccoreParallelFor(host_bounds, launch_info.loopRunInfo(), func, other_args...);
  } break;
  default:
    ARCCORE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to retain the arguments of a RunCommand.
 */
template <typename LoopBoundType, typename... RemainingArgs>
class ExtendedLaunchRunCommand
{
 public:

  ExtendedLaunchRunCommand(RunCommand& command, const LoopBoundType& bounds)
  : m_command(command)
  , m_bounds(bounds)
  {
  }
  ExtendedLaunchRunCommand(RunCommand& command, const LoopBoundType& bounds, const std::tuple<RemainingArgs...>& args)
  : m_command(command)
  , m_bounds(bounds)
  , m_remaining_args(args)
  {
  }
  RunCommand& m_command;
  LoopBoundType m_bounds;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to manage the launch of a hierarchical compute kernel.
 */
template <typename LoopBoundType, typename... RemainingArgs>
class ExtendedLaunchLoop
{
 public:

  ExtendedLaunchLoop(const LoopBoundType& bounds, RemainingArgs... args)
  : m_bounds(bounds)
  , m_remaining_args(args...)
  {
  }
  LoopBoundType m_bounds;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename... RemainingArgs> auto
makeLaunch(const LoopBoundType& bounds, RemainingArgs... args)
-> ExtendedLaunchLoop<LoopBoundType, RemainingArgs...>
{
  return ExtendedLaunchLoop<LoopBoundType, RemainingArgs...>(bounds, args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
operator<<(ExtendedLaunchRunCommand<LoopBoundType, RemainingArgs...>&& nr, const Lambda& f)
{
  if constexpr (sizeof...(RemainingArgs) > 0) {
    std::apply([&](auto... vs) { _doHierarchicalLaunch(nr.m_command, nr.m_bounds, f, vs...); }, nr.m_remaining_args);
  }
  else {
    _doHierarchicalLaunch(nr.m_command, nr.m_bounds, f);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Applies the functor \a func on a sequential loop.
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
arccoreSequentialFor(HostLaunchLoopRange<LoopBoundType> bounds, const Lambda& func, const RemainingArgs&... remaining_args)
{
  WorkGroupSequentialForHelper::apply(0, bounds.nbBlock(), bounds, func, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Applies the functor \a func on a parallel loop.
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
arccoreParallelFor(HostLaunchLoopRange<LoopBoundType> bounds, ForLoopRunInfo run_info,
                   const Lambda& func, const RemainingArgs&... remaining_args)
{
  Int32 nb_thread = run_info.options().value().maxThread();
  ThreadGridSynchronizer grid_sync(nb_thread);
  bounds.setThreadGridSynchronizer(&grid_sync);
  auto sub_func = [=](Int32 begin_index, Int32 nb_loop) {
    Impl::WorkGroupSequentialForHelper::apply(begin_index, nb_loop, bounds, func, remaining_args...);
  };
  ::Arcane::arccoreParallelFor(0, bounds.nbBlock(), run_info, sub_func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
