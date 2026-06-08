// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.h                                              (C) 2000-2026 */
/*                                                                           */
/* Implementation of a 'RunQueue'.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNQUEUEIMPL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNQUEUEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include "arccore/common/Array.h"

#include <stack>
#include <atomic>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Execution queue for accelerator.
 *
 * This class manages the implementation of a RunQueue.
 * The method descriptions are in RunQueue.
 */
class ARCCORE_COMMON_EXPORT RunQueueImpl
{
  friend class Arcane::Accelerator::Runner;
  friend class Arcane::Accelerator::RunQueue;
  friend class RunCommandImpl;
  friend class RunQueueImplStack;
  friend class RunnerImpl;

  class Lock;

 private:

  RunQueueImpl(RunnerImpl* runner_impl, Int32 id, const RunQueueBuildInfo& bi);

 private:

  // You must use _destroy() to destroy the instance.
  ~RunQueueImpl();

 public:

  RunQueueImpl(const RunQueueImpl&) = delete;
  RunQueueImpl(RunQueueImpl&&) = delete;
  RunQueueImpl& operator=(const RunQueueImpl&) = delete;
  RunQueueImpl& operator=(RunQueueImpl&&) = delete;

 public:

  static RunQueueImpl* create(RunnerImpl* r, const RunQueueBuildInfo& bi);
  static RunQueueImpl* create(RunnerImpl* r);

 public:

  eExecutionPolicy executionPolicy() const { return m_execution_policy; }
  RunnerImpl* runner() const { return m_runner_impl; }
  MemoryAllocationOptions allocationOptions() const;
  bool isAutoPrefetchCommand() const;

  void copyMemory(const MemoryCopyArgs& args) const;
  void prefetchMemory(const MemoryPrefetchArgs& args) const;

  void recordEvent(RunQueueEvent& event);
  void waitEvent(RunQueueEvent& event);

  void setConcurrentCommandCreation(bool v);
  bool isConcurrentCommandCreation() const { return m_use_pool_mutex; }

  void dumpStats(std::ostream& ostr) const;
  bool isAsync() const { return m_is_async; }

  void _internalBarrier();
  IRunnerRuntime* _internalRuntime() const { return m_runtime; }
  IRunQueueStream* _internalStream() const { return m_queue_stream; }

 public:

  void addRef()
  {
    ++m_nb_ref;
  }
  void removeRef()
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
      _release();
  }

 private:

  RunCommandImpl* _internalCreateOrGetRunCommandImpl();
  void _internalFreeRunningCommands();
  bool _isInPool() const { return m_is_in_pool; }
  void _release();
  void _setDefaultMemoryRessource();
  void _addRunningCommand(RunCommandImpl* p);
  void _putInCommandPool(RunCommandImpl* p);
  void _freeCommandsInPool();
  void _checkPutCommandInPoolNoLock(RunCommandImpl* p);
  static RunQueueImpl* _reset(RunQueueImpl* p);
  static void _destroy(RunQueueImpl* q);

 private:

  RunnerImpl* m_runner_impl = nullptr;
  eExecutionPolicy m_execution_policy = eExecutionPolicy::None;
  IRunnerRuntime* m_runtime = nullptr;
  IRunQueueStream* m_queue_stream = nullptr;
  //! Command pool
  std::stack<RunCommandImpl*> m_run_command_pool;
  //! List of running commands
  UniqueArray<RunCommandImpl*> m_active_run_command_list;
  //! Queue ID
  Int32 m_id = 0;
  //! Indicates if the instance is in an instance pool.
  bool m_is_in_pool = false;
  //! Number of references on the instance.
  std::atomic<Int32> m_nb_ref = 0;
  //! Indicates if the queue is asynchronous
  bool m_is_async = false;
  //! Default memory resource
  eMemoryResource m_memory_ressource = eMemoryResource::Unknown;

  // Mutex for commands (active if \a m_use_pool_mutex is true)
  std::unique_ptr<std::mutex> m_pool_mutex;
  bool m_use_pool_mutex = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
