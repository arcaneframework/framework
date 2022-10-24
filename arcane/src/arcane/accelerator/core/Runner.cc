// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.cc                                                   (C) 2000-2022 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/Runner.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/accelerator/core/RunQueueImpl.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/IRunQueueRuntime.h"
#include "arcane/accelerator/core/DeviceId.h"

#include <stack>
#include <map>
#include <atomic>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

namespace
{
  inline impl::IRunQueueRuntime*
  _getRuntime(eExecutionPolicy p)
  {
    impl::IRunQueueRuntime* runtime = nullptr;
    switch (p) {
    case eExecutionPolicy::None:
      ARCANE_FATAL("No runtime for eExecutionPolicy::None");
    case eExecutionPolicy::HIP:
      return impl::getHIPRunQueueRuntime();
    case eExecutionPolicy::CUDA:
      return impl::getCUDARunQueueRuntime();
    case eExecutionPolicy::Sequential:
      return impl::getSequentialRunQueueRuntime();
    case eExecutionPolicy::Thread:
      return impl::getThreadRunQueueRuntime();
      ;
    }
    return runtime;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Runner::Impl
{
  class RunQueueImplStack
  {
   public:

    RunQueueImplStack(Runner* runner, eExecutionPolicy exec_policy, impl::IRunQueueRuntime* runtime)
    : m_runner(runner)
    , m_exec_policy(exec_policy)
    , m_runtime(runtime)
    {}

   public:

    bool empty() const { return m_stack.empty(); }
    void pop() { m_stack.pop(); }
    impl::RunQueueImpl* top() { return m_stack.top(); }
    void push(impl::RunQueueImpl* v) { m_stack.push(v); }

   public:

    impl::RunQueueImpl* createRunQueue(const RunQueueBuildInfo& bi)
    {
      // Si pas de runtime, essaie de le récupérer. On le fait ici et aussi
      // lors de la création de l'instance car l'utilisateur a pu ajouter une
      // implémentation de runtime entre-temps (par exemple si l'instance de Runner
      // a été créée avant l'initialisation du runtime).
      if (!m_runtime)
        m_runtime = _getRuntime(m_exec_policy);
      if (!m_runtime)
        ARCANE_FATAL("Can not create RunQueue for execution policy '{0}' "
                     "because no RunQueueRuntime is available for this policy",
                     m_exec_policy);
      Int32 x = ++m_nb_created;
      auto* q = new impl::RunQueueImpl(m_runner, x, m_runtime, bi);
      q->m_is_in_pool = true;
      return q;
    }

   private:

    std::stack<impl::RunQueueImpl*> m_stack;
    std::atomic<Int32> m_nb_created = -1;
    Runner* m_runner;
    eExecutionPolicy m_exec_policy;
    impl::IRunQueueRuntime* m_runtime;
  };

 public:

  //! Verrou pour le pool de RunQueue en multi-thread.
  class Lock
  {
   public:

    explicit Lock(Impl* p)
    {
      if (p->m_use_pool_mutex) {
        m_mutex = p->m_pool_mutex.get();
        if (m_mutex)
          m_mutex->lock();
      }
    }
    ~Lock()
    {
      if (m_mutex)
        m_mutex->unlock();
    }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

   private:

    std::mutex* m_mutex = nullptr;
  };

 public:

  ~Impl()
  {
    for (auto& x : m_run_queue_pool_map) {
      _freePool(x.second);
      delete x.second;
    }
  }

 public:

  void build(Runner* runner)
  {
    _add(runner, eExecutionPolicy::Sequential);
    _add(runner, eExecutionPolicy::Thread);
    _add(runner, eExecutionPolicy::CUDA);
    _add(runner, eExecutionPolicy::HIP);
  }
  void setConcurrentQueueCreation(bool v)
  {
    m_use_pool_mutex = v;
    if (!m_pool_mutex.get())
      m_pool_mutex = std::make_unique<std::mutex>();
  }
  bool isConcurrentQueueCreation() const { return m_use_pool_mutex; }

 public:

  RunQueueImplStack* getPool(eExecutionPolicy exec_policy)
  {
    auto x = m_run_queue_pool_map.find(exec_policy);
    if (x == m_run_queue_pool_map.end())
      ARCANE_FATAL("No RunQueueImplStack for execution policy '{0}'", (int)exec_policy);
    return x->second;
  }
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

 public:

  //TODO: mettre à None lorsqu'on aura supprimé Runner::setExecutionPolicy()
  eExecutionPolicy m_execution_policy = eExecutionPolicy::Sequential;
  bool m_is_init = false;
  eDeviceReducePolicy m_reduce_policy = eDeviceReducePolicy::Atomic;
  DeviceId m_device_id;

 private:

  std::map<eExecutionPolicy, RunQueueImplStack*> m_run_queue_pool_map;
  std::unique_ptr<std::mutex> m_pool_mutex;
  bool m_use_pool_mutex = false;
  /*!
   * \brief Temps passé dans le noyau en nano-seconde. On utilise un 'Int64'
   * car les atomiques sur les flottants ne sont pas supportés partout.
   */
  std::atomic<Int64> m_cumulative_command_time = 0;

 private:

  void _freePool(RunQueueImplStack* s)
  {
    while (!s->empty()) {
      delete s->top();
      s->pop();
    }
  }
  void _add(Runner* runner, eExecutionPolicy exec_policy)
  {
    impl::IRunQueueRuntime* r = _getRuntime(exec_policy);
    auto* q = new RunQueueImplStack(runner, exec_policy, r);
    m_run_queue_pool_map.insert(std::make_pair(exec_policy, q));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
Runner()
: m_p(new Impl())
{
  m_p->build(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
Runner(eExecutionPolicy p)
: Runner()
{
  initialize(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
~Runner()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunQueueImpl* Runner::
_internalCreateOrGetRunQueueImpl(eExecutionPolicy exec_policy)
{
  _checkIsInit();

  auto pool = m_p->getPool(exec_policy);

  {
    Impl::Lock my_lock(m_p);
    if (!pool->empty()) {
      impl::RunQueueImpl* p = pool->top();
      pool->pop();
      return p;
    }
  }

  return pool->createRunQueue(RunQueueBuildInfo{});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunQueueImpl* Runner::
_internalCreateOrGetRunQueueImpl(const RunQueueBuildInfo& bi)
{
  _checkIsInit();
  // Si on utilise les paramètres par défaut, on peut utilier une RunQueueImpl
  // issue du pool.
  eExecutionPolicy p = executionPolicy();
  if (bi.isDefault())
    return _internalCreateOrGetRunQueueImpl(p);
  impl::IRunQueueRuntime* runtime = _getRuntime(p);
  ARCANE_CHECK_POINTER(runtime);
  auto* queue = new impl::RunQueueImpl(this, 0, runtime, bi);
  return queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
_internalFreeRunQueueImpl(impl::RunQueueImpl* p)
{
  _checkIsInit();
  {
    Impl::Lock my_lock(m_p);
    if (p->_isInPool())
      m_p->getPool(p->executionPolicy())->push(p);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eExecutionPolicy Runner::
executionPolicy() const
{
  return m_p->m_execution_policy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Runner::
isInitialized() const
{
  return m_p->m_is_init;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
setConcurrentQueueCreation(bool v)
{
  m_p->setConcurrentQueueCreation(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Runner::
isConcurrentQueueCreation() const
{
  return m_p->isConcurrentQueueCreation();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
setDeviceReducePolicy(eDeviceReducePolicy v)
{
  m_p->m_reduce_policy = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eDeviceReducePolicy Runner::
deviceReducePolicy() const
{
  return m_p->m_reduce_policy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunQueueEventImpl* Runner::
_createEvent()
{
  _checkIsInit();
  impl::IRunQueueRuntime* r = _getRuntime(executionPolicy());
  return r->createEventImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunQueueEventImpl* Runner::
_createEventWithTimer()
{
  _checkIsInit();
  impl::IRunQueueRuntime* r = _getRuntime(executionPolicy());
  return r->createEventImplWithTimer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
initialize(eExecutionPolicy v)
{
  if (m_p->m_is_init)
    ARCANE_FATAL("Runner is already initialized");
  if (v == eExecutionPolicy::None)
    ARCANE_THROW(ArgumentException, "executionPolicy should not be eExecutionPolicy::None");
  m_p->m_execution_policy = v;
  m_p->m_is_init = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
_checkIsInit() const
{
  if (!m_p->m_is_init)
    ARCANE_FATAL("Runner is not initialized. Call method initialize() before");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
_addCommandTime(double v)
{
  m_p->addTime(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

double Runner::
cumulativeCommandTime() const
{
  return m_p->cumulativeCommandTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
setMemoryAdvice(MemoryView buffer, eMemoryAdvice advice)
{
  impl::IRunQueueRuntime* r = _getRuntime(executionPolicy());
  r->setMemoryAdvice(buffer, advice, m_p->m_device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
unsetMemoryAdvice(MemoryView buffer, eMemoryAdvice advice)
{
  impl::IRunQueueRuntime* r = _getRuntime(executionPolicy());
  r->unsetMemoryAdvice(buffer, advice, m_p->m_device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
