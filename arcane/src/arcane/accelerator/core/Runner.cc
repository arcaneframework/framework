// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.cc                                                   (C) 2000-2023 */
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

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/accelerator/core/IDeviceInfoList.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"

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
  inline impl::IRunnerRuntime*
  _getRuntimeNoCheck(eExecutionPolicy p)
  {
    impl::IRunnerRuntime* runtime = nullptr;
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
    }
    return runtime;
  }

  inline impl::IRunnerRuntime*
  _getRuntime(eExecutionPolicy p)
  {
    auto* x = _getRuntimeNoCheck(p);
    if (!x)
      ARCANE_FATAL("No runtime is available for execution policy '{0}'", p);
    return x;
  }

  inline void
  _stopProfiling(eExecutionPolicy p)
  {
    auto* x = _getRuntimeNoCheck(p);
    if (x)
      x->stopProfiling();
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Runner::Impl
{
  class RunQueueImplStack
  {
   public:

    explicit RunQueueImplStack(Runner* runner)
    : m_runner(runner)
    {}

   public:

    bool empty() const { return m_stack.empty(); }
    void pop() { m_stack.pop(); }
    impl::RunQueueImpl* top() { return m_stack.top(); }
    void push(impl::RunQueueImpl* v) { m_stack.push(v); }

   public:

    impl::RunQueueImpl* createRunQueue(const RunQueueBuildInfo& bi)
    {
      Int32 x = ++m_nb_created;
      auto* q = new impl::RunQueueImpl(m_runner, x, bi);
      q->m_is_in_pool = true;
      return q;
    }

   private:

    std::stack<impl::RunQueueImpl*> m_stack;
    std::atomic<Int32> m_nb_created = -1;
    Runner* m_runner;
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
    _freePool(m_run_queue_pool);
    delete m_run_queue_pool;
  }

 public:

  void initialize(Runner* runner, eExecutionPolicy v, DeviceId device)
  {
    if (m_is_init)
      ARCANE_FATAL("Runner is already initialized");
    if (v == eExecutionPolicy::None)
      ARCANE_THROW(ArgumentException, "executionPolicy should not be eExecutionPolicy::None");
    if (device.isHost() || device.isNull())
      ARCANE_THROW(ArgumentException, "device should not be Device::hostDevice() or Device::nullDevice()");

    m_execution_policy = v;
    m_device_id = device;
    m_runtime = _getRuntime(v);
    m_is_init = true;

    // Il faut initialiser le pool à la fin car il a besoin d'accéder à \a m_runtime
    m_run_queue_pool = new RunQueueImplStack(runner);
  }

  void setConcurrentQueueCreation(bool v)
  {
    m_use_pool_mutex = v;
    if (!m_pool_mutex.get())
      m_pool_mutex = std::make_unique<std::mutex>();
  }
  bool isConcurrentQueueCreation() const { return m_use_pool_mutex; }

 public:

  RunQueueImplStack* getPool()
  {
    if (!m_run_queue_pool)
      ARCANE_FATAL("Runner is not initialized");
    return m_run_queue_pool;
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

  impl::IRunnerRuntime* runtime() const { return m_runtime; }

 public:

  //TODO: mettre à None lorsqu'on aura supprimé Runner::setExecutionPolicy()
  eExecutionPolicy m_execution_policy = eExecutionPolicy::None;
  bool m_is_init = false;
  eDeviceReducePolicy m_reduce_policy = eDeviceReducePolicy::Grid;
  DeviceId m_device_id;

 private:

  impl::IRunnerRuntime* m_runtime = nullptr;
  RunQueueImplStack* m_run_queue_pool = nullptr;
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
    if (!s)
      return;
    while (!s->empty()) {
      delete s->top();
      s->pop();
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
Runner()
: m_p(std::make_shared<Impl>())
{
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
Runner(eExecutionPolicy p, DeviceId device_id)
: Runner()
{
  initialize(p, device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunnerRuntime* Runner::
_internalRuntime() const
{
  return m_p->runtime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunQueueImpl* Runner::
_internalCreateOrGetRunQueueImpl()
{
  _checkIsInit();

  auto pool = m_p->getPool();

  {
    Impl::Lock my_lock(m_p.get());
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
  if (bi.isDefault())
    return _internalCreateOrGetRunQueueImpl();
  impl::IRunnerRuntime* runtime = m_p->runtime();
  ARCANE_CHECK_POINTER(runtime);
  auto* queue = new impl::RunQueueImpl(this, 0, bi);
  return queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
_internalPutRunQueueImplInPool(impl::RunQueueImpl* p)
{
  _checkIsInit();
  {
    Impl::Lock my_lock(m_p.get());
    m_p->getPool()->push(p);
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
  return m_p->runtime()->createEventImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunQueueEventImpl* Runner::
_createEventWithTimer()
{
  _checkIsInit();
  return m_p->runtime()->createEventImplWithTimer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
initialize(eExecutionPolicy v)
{
  m_p->initialize(this, v, DeviceId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
initialize(eExecutionPolicy v, DeviceId device_id)
{
  m_p->initialize(this, v, device_id);
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
setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice)
{
  _checkIsInit();
  m_p->runtime()->setMemoryAdvice(buffer, advice, m_p->m_device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice)
{
  _checkIsInit();
  m_p->runtime()->unsetMemoryAdvice(buffer, advice, m_p->m_device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
setAsCurrentDevice()
{
  _checkIsInit();
  m_p->runtime()->setCurrentDevice(m_p->m_device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DeviceId Runner::
deviceId() const
{
  return m_p->m_device_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const DeviceInfo& Runner::
deviceInfo() const
{
  _checkIsInit();
  const IDeviceInfoList* dlist = deviceInfoList(executionPolicy());
  Int32 nb_device = dlist->nbDevice();
  if (nb_device == 0)
    ARCANE_FATAL("Internal error: no device available");
  Int32 device_id = deviceId().asInt32();
  if (device_id >= nb_device)
    device_id = 0;
  return dlist->deviceInfo(device_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const IDeviceInfoList* Runner::
deviceInfoList(eExecutionPolicy policy)
{
  if (policy == eExecutionPolicy::None)
    return nullptr;
  impl::IRunnerRuntime* r = _getRuntime(policy);
  return r->deviceInfoList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
fillPointerAttribute(PointerAttribute& attr, const void* ptr)
{
  _checkIsInit();
  m_p->runtime()->getPointerAttribute(attr, ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arrête tout les profiling en cours de tout les runtime.
 *
 * En général on utilise cela en fin de calcul.
 */
void Runner::
stopAllProfiling()
{
  _stopProfiling(eExecutionPolicy::CUDA);
  _stopProfiling(eExecutionPolicy::HIP);
  _stopProfiling(eExecutionPolicy::Sequential);
  _stopProfiling(eExecutionPolicy::Thread);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ePointerAccessibility
getPointerAccessibility(Runner* runner, const void* ptr, PointerAttribute* ptr_attr)
{
  if (!runner)
    return ePointerAccessibility::Unknown;
  return impl::RuntimeStaticInfo::getPointerAccessibility(runner->executionPolicy(), ptr, ptr_attr);
}

extern "C++" void impl::
arcaneCheckPointerIsAcccessible(Runner* runner, const void* ptr,
                                const char* name, const TraceInfo& ti)
{
  if (!runner)
    return;
  return impl::RuntimeStaticInfo::checkPointerIsAcccessible(runner->executionPolicy(), ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::IRunnerRuntime::
_fillPointerAttribute(PointerAttribute& attribute,
                      ePointerMemoryType mem_type,
                      int device, const void* pointer, const void* device_pointer,
                      const void* host_pointer)
{
  attribute = PointerAttribute(mem_type, device, pointer, device_pointer, host_pointer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::IRunnerRuntime::
_fillPointerAttribute(PointerAttribute& attribute, const void* pointer)
{
  attribute = PointerAttribute(pointer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
