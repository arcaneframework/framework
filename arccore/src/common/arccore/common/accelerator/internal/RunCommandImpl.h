// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandImpl.h                                            (C) 2000-2026 */
/*                                                                           */
/* Implementation of command management on accelerator.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_RUNCOMMANDIMPL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_RUNCOMMANDIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/TraceInfo.h"
#include "arccore/base/Profiling.h"
#include "arccore/base/String.h"
#include "arccore/base/ParallelLoopOptions.h"

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include <set>
#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Implementation of a command for accelerator.
 */
class RunCommandImpl
{
  friend RunCommand;
  friend RunQueueImpl;

 public:

  explicit RunCommandImpl(RunQueueImpl* queue);
  ~RunCommandImpl();
  RunCommandImpl(const RunCommandImpl&) = delete;
  RunCommandImpl& operator=(const RunCommandImpl&) = delete;

 public:

  static RunCommandImpl* create(RunQueueImpl* r);

 public:

  const TraceInfo& traceInfo() const { return m_trace_info; }
  const String& kernelName() const { return m_kernel_name; }

 public:

  void notifyBeginLaunchKernel();
  void notifyEndLaunchKernel();
  void notifyEndExecuteKernel();
  Impl::IReduceMemoryImpl* getOrCreateReduceMemoryImpl();
  void releaseReduceMemoryImpl(ReduceMemoryImpl* p);
  IRunQueueStream* internalStream() const;
  RunnerImpl* runner() const;
  bool hasActiveReduction() const { return !m_active_reduce_memory_list.empty(); }

 public:

  void notifyLaunchKernelSyclEvent(void* sycl_event_ptr);

 private:

  ReduceMemoryImpl* _getOrCreateReduceMemoryImpl();

 private:

  RunQueueImpl* m_queue;
  TraceInfo m_trace_info;
  String m_kernel_name;
  Int32 m_nb_thread_per_block = 0;
  ParallelLoopOptions m_parallel_loop_options;

  // NOTE: this stack manages the memory associated with a single runtime
  // If we ever want to support multiple runtimes, a stack will be needed
  // per runtime. We can possibly limit this if we are sure
  // that a command is associated with only one type (in the runtime sense) of RunQueue.
  std::stack<ReduceMemoryImpl*> m_reduce_memory_pool;

  //! List of active reductions
  std::set<ReduceMemoryImpl*> m_active_reduce_memory_list;

  //! Indicates if the command has been launched.
  bool m_has_been_launched = false;

  //! Indicates if profiling is desired
  bool m_use_profiling = false;

  //! Indicates if sequential events are used to calculate execution time
  bool m_use_sequential_timer_event = false;

  //! Events for the start and end of execution.
  IRunQueueEventImpl* m_start_event = nullptr;
  //! Events for the end of execution.
  IRunQueueEventImpl* m_stop_event = nullptr;

  //! Time when the command is launched
  Int64 m_begin_time = 0;

  ForLoopOneExecStat m_loop_one_exec_stat;
  ForLoopOneExecStat* m_loop_one_exec_stat_ptr = nullptr;

  //! Indicates if the command executes on the accelerator
  const eExecutionPolicy m_execution_policy = eExecutionPolicy::None;

  //! Indicates if the command executes on the accelerator
  const bool m_use_accelerator = false;

  /*!
   * \brief Indicates if we allow the same command to be used multiple times.
   *
   * Normally this is forbidden, but before November 2024, there was no
   * mechanism to detect this. We can therefore temporarily allow
   * this, and we will remove this possibility in a future version.
   */
  bool m_is_allow_reuse_command = false;

  //! Indicates if a RunCommand has a reference to this instance.
  bool m_has_living_run_command = false;

  //! Indicates if the command can be returned to the pool associated with the RunQueue.
  bool m_may_be_put_in_pool = false;

  //! Size of the shared memory to allocate
  Int32 m_shared_memory_size = 0;

  //! Number of loop decomposition strides
  Int32 m_nb_stride = 1;

  //! Default number of decomposition strides
  Int32 m_default_nb_stride = 1;

 private:

  void _freePools();
  void _reset();
  void _init();
  IRunQueueEventImpl* _createEvent();
  void _notifyDestroyRunCommand();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
