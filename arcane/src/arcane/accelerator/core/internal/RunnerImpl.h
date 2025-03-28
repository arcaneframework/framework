// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunnerImpl.h                                                (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'un 'Runner'.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_RUNNERIMPL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_RUNNERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/accelerator/core/internal/RunnerInternal.h"

#include <stack>
#include <mutex>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RunQueueImplStack
{
 public:

  void initialize(RunnerImpl* v) { m_runner_impl = v; }

 public:

  bool empty() const { return m_stack.empty(); }
  void pop() { m_stack.pop(); }
  impl::RunQueueImpl* top() { return m_stack.top(); }
  void push(impl::RunQueueImpl* v) { m_stack.push(v); }

 public:

  RunQueueImpl* createRunQueue(const RunQueueBuildInfo& bi);

 private:

  std::stack<impl::RunQueueImpl*> m_stack;
  std::atomic<Int32> m_nb_created = -1;
  RunnerImpl* m_runner_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RunnerImpl
{
  friend ::Arcane::Accelerator::Runner;

 public:

  //! Verrou pour le pool de RunQueue en multi-thread.
  class Lock
  {
   public:

    explicit Lock(RunnerImpl* p)
    : m_mutex_ptr(&p->m_pool_mutex)
    {
      m_mutex_ptr->lock();
    }
    ~Lock()
    {
      m_mutex_ptr->unlock();
    }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

   private:

    std::mutex* m_mutex_ptr = nullptr;
  };

 public:

  RunnerImpl()
  : m_internal_api(this)
  {
  }

  ~RunnerImpl()
  {
    _freePool();
  }

 public:

  void initialize(Runner* runner, eExecutionPolicy v, DeviceId device);

 public:

  RunQueueImplStack* getPool();
  void addTime(double v)
  {
    // 'v' est en seconde. On le convertit en nanosecond.
    Int64 x = static_cast<Int64>(v * 1.0e9);
    m_cumulative_command_time += x;
  }
  double cumulativeCommandTime() const
  {
    Int64 x = m_cumulative_command_time;
    return static_cast<double>(x) / 1.0e9;
  }

  impl::IRunnerRuntime* runtime() const { return m_runtime; }
  bool isAutoPrefetchCommand() const { return m_is_auto_prefetch_command; }

  eExecutionPolicy executionPolicy() const { return m_execution_policy; }
  bool isInit() const { return m_is_init; }
  eDeviceReducePolicy reducePolicy() const { return m_reduce_policy; }
  DeviceId deviceId() const { return m_device_id; }

 public:

  void _internalPutRunQueueImplInPool(RunQueueImpl* p);
  RunQueueImpl* _internalCreateOrGetRunQueueImpl();
  RunQueueImpl* _internalCreateOrGetRunQueueImpl(const RunQueueBuildInfo& bi);
  IRunQueueEventImpl* _createEvent();
  IRunQueueEventImpl* _createEventWithTimer();
  RunnerInternal* _internalApi() { return &m_internal_api; }

 private:

  eExecutionPolicy m_execution_policy = eExecutionPolicy::None;
  bool m_is_init = false;
  const eDeviceReducePolicy m_reduce_policy = eDeviceReducePolicy::Grid;
  DeviceId m_device_id;
  impl::IRunnerRuntime* m_runtime = nullptr;
  RunQueueImplStack m_run_queue_pool;
  std::mutex m_pool_mutex;
  /*!
   * \brief Temps passé dans le noyau en nano-seconde. On utilise un 'Int64'
   * car les atomiques sur les flottants ne sont pas supportés partout.
   */
  std::atomic<Int64> m_cumulative_command_time = 0;

  //! Indique si on pré-copie les données avant une commande de cette RunQueue
  bool m_is_auto_prefetch_command = false;

  RunnerInternal m_internal_api;

 private:

  void _freePool();
  void _checkIsInit() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
