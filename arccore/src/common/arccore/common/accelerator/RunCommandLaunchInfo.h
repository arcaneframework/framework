// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.h                                      (C) 2000-2026 */
/*                                                                           */
/* Information for running a 'RunCommand'.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNCOMMANDLAUNCHINFO_H
#define ARCCORE_COMMON_ACCELERATOR_RUNCOMMANDLAUNCHINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Profiling.h"
#include "arccore/base/ForLoopRunInfo.h"

#include "arccore/common/accelerator/KernelLaunchArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Temporary object to store the execution information of a
 * command and group tests.
 */
class ARCCORE_COMMON_EXPORT RunCommandLaunchInfo
{
  // The following classes allow kernels to be launched.
  friend class CudaKernelLauncher;
  friend class HipKernelLauncher;
  friend class SyclKernelLauncher;

 public:

  using ThreadBlockInfo = KernelLaunchArgs;

 public:

  RunCommandLaunchInfo(RunCommand& command, Int64 total_loop_size);
  RunCommandLaunchInfo(RunCommand& command, Int64 total_loop_size, bool is_cooperative);
  ~RunCommandLaunchInfo() noexcept(false);
  RunCommandLaunchInfo(const RunCommandLaunchInfo&) = delete;
  RunCommandLaunchInfo operator=(const RunCommandLaunchInfo&) = delete;

 public:

  eExecutionPolicy executionPolicy() const { return m_exec_policy; }

  /*!
   * \brief Indicates that command execution is starting.
   *
   * Must always be called before launching the command to ensure that
   * this method is called in case of an exception.
   */
  void beginExecute();

  /*!
   * \brief Signals the end of execution.
   *
   * If the queue associated with the command is asynchronous, the command
   * may continue to execute after this call.
   */
  void endExecute();

  //! Calculates and returns the information for multi-thread loops
  ParallelLoopOptions computeParallelLoopOptions() const;

  /*!
   * \brief Loop execution information.
   *
   * This information is only valid if executionPolicy()==eExecutionPolicy::Thread
   * and if beginExecute() has been called.
   */
  const ForLoopRunInfo& loopRunInfo() const { return m_loop_run_info; }

  //! Total loop size
  Int64 totalLoopSize() const { return m_total_loop_size; }

 private:

  RunCommand& m_command;
  bool m_has_exec_begun = false;
  bool m_is_notify_end_kernel_done = false;
  bool m_is_need_barrier = false;
  bool m_is_forced_need_barrier = false;
  bool m_is_cooperative_launch = false;
  eExecutionPolicy m_exec_policy = eExecutionPolicy::Sequential;
  KernelLaunchArgs m_kernel_launch_args;
  ForLoopRunInfo m_loop_run_info;
  Int64 m_total_loop_size = 0;
  RunQueueImpl* m_queue_impl = nullptr;

 private:

  //! Calculates the arguments for launching the kernel whose address is \a func
  KernelLaunchArgs _computeKernelLaunchArgs(const void* func) const;
  NativeStream _internalNativeStream();
  void _doEndKernelLaunch();
  void _computeInitialKernelLaunchArgs();

  // For testing only with CUDA
  bool _isUseCooperativeLaunch() const;
  bool _isUseCudaLaunchKernel() const;
  void _setIsNeedBarrier(bool v);

 private:

  void _computeLoopRunInfo();

  // For SYCL: registers the event associated with the last command in the queue
  // \a sycl_event_ptr is of type 'sycl::event*'.
  void _addSyclEvent(void* sycl_event_ptr);
  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
