// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Information for running a 'RunCommand'.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/CheckedConvert.h"
#include "arccore/base/ConcurrencyBase.h"

#include "arccore/common/accelerator/KernelLaunchArgs.h"
#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/common/accelerator/NativeStream.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/RunCommandImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_init()
{
  m_queue_impl = m_command._internalQueueImpl();
  m_exec_policy = m_queue_impl->executionPolicy();
  // Kernel launch information calculation is only useful on accelerator
  if (isAcceleratorPolicy(m_exec_policy)) {
    _computeInitialKernelLaunchArgs();
    m_command._allocateReduceMemory(m_kernel_launch_args.nbBlockPerGrid());
    // If reductions are present, we force the barrier at the end of the kernel
    m_is_forced_need_barrier = m_command.m_p->hasActiveReduction();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
RunCommandLaunchInfo(RunCommand& command, Int64 total_loop_size)
: m_command(command)
, m_total_loop_size(total_loop_size)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
RunCommandLaunchInfo(RunCommand& command, Int64 total_loop_size, bool is_cooperative)
: m_command(command)
, m_is_cooperative_launch(is_cooperative)
, m_total_loop_size(total_loop_size)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
~RunCommandLaunchInfo() noexcept(false)
{
  // Notifies the end of kernel launch. Normally, this is already done
  // unless there was an exception during the computation kernel launch.
  _doEndKernelLaunch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
beginExecute()
{
  if (m_has_exec_begun)
    ARCCORE_FATAL("beginExecute() has already been called");
  m_has_exec_begun = true;
  m_command._internalNotifyBeginLaunchKernel();
  if (m_exec_policy == eExecutionPolicy::Thread)
    _computeLoopRunInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Notifies the end of command launch.
 *
 * Note that if the command is asynchronous, its execution may continue
 * after calling this method.
 */
void RunCommandLaunchInfo::
endExecute()
{
  if (!m_has_exec_begun)
    ARCCORE_FATAL("beginExecute() has to be called before endExecute()");
  _doEndKernelLaunch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_doEndKernelLaunch()
{
  if (m_is_notify_end_kernel_done)
    return;
  m_is_notify_end_kernel_done = true;
  m_command._internalNotifyEndLaunchKernel();

  Impl::RunQueueImpl* q = m_queue_impl;
  if (!q->isAsync() || m_is_need_barrier || m_is_forced_need_barrier)
    q->_internalBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NativeStream RunCommandLaunchInfo::
_internalNativeStream()
{
  return m_command._internalNativeStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the initial value of kernel block/thread/grid
 * based on \a full_size.
 */
void RunCommandLaunchInfo::
_computeInitialKernelLaunchArgs()
{
  int threads_per_block = m_command.nbThreadPerBlock();
  if (threads_per_block<=0)
    threads_per_block = 256;
  Int64 big_b = (m_total_loop_size + threads_per_block - 1) / threads_per_block;
  int blocks_per_grid = CheckedConvert::toInt32(big_b);
  m_kernel_launch_args = KernelLaunchArgs(blocks_per_grid, threads_per_block);
  m_kernel_launch_args.setSharedMemorySize(m_command._sharedMemory());
  m_kernel_launch_args.setIsCooperative(m_is_cooperative_launch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelLoopOptions RunCommandLaunchInfo::
computeParallelLoopOptions() const
{
  ParallelLoopOptions opt = m_command.parallelLoopOptions();
  const bool use_dynamic_compute = true;
  // Calculates a default grain size if it is not specified in
  // the options. By default, we ensure a number of iterations
  // equal to 2 times the number of threads used.
  if (use_dynamic_compute && opt.grainSize() == 0) {
    Int32 nb_thread = opt.maxThread();
    if (nb_thread <= 0)
      nb_thread = ConcurrencyBase::maxAllowedThread();
    if (nb_thread <= 0)
      nb_thread = 1;
    Int32 grain_size = static_cast<Int32>((double)m_total_loop_size / (nb_thread * 2.0));
    opt.setGrainSize(grain_size);
  }
  return opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the value of m_loop_run_info.
 *
 * This is only useful in multi-thread mode.
 */
void RunCommandLaunchInfo::
_computeLoopRunInfo()
{
  ForLoopTraceInfo lti(m_command.traceInfo(), m_command.kernelName());
  m_loop_run_info = ForLoopRunInfo(computeParallelLoopOptions(), lti);
  m_loop_run_info.setExecStat(m_command._internalCommandExecStat());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Determines the kernel configuration.
 *
 * The configuration depends on the underlying runtime. For CUDA and ROCM,
 * it is a number of blocks and threads.
 *
 * It is possible to dynamically calculate the optimal values to
 * maximize occupancy.
 */
KernelLaunchArgs RunCommandLaunchInfo::
_computeKernelLaunchArgs(const void* func) const
{
  Impl::IRunnerRuntime* r = m_queue_impl->_internalRuntime();

  return r->computeKernalLaunchArgs(m_kernel_launch_args, func,
                                    totalLoopSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_addSyclEvent(void* sycl_event_ptr)
{
  m_command._internalNotifyBeginLaunchKernelSyclEvent(sycl_event_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunCommandLaunchInfo::
_isUseCooperativeLaunch() const
{
  // Indicates if cudaLaunchCooperativeKernel() is used
  return m_is_cooperative_launch;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunCommandLaunchInfo::
_isUseCudaLaunchKernel() const
{
  // Indicates if cudaLaunchKernel() is used instead of kernel<<<...>>>.
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_setIsNeedBarrier(bool v)
{
  m_is_need_barrier = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
