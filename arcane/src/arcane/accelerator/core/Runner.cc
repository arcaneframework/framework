// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.cc                                                   (C) 2000-2024 */
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
#include "arcane/utils/ValueConvert.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/accelerator/core/IDeviceInfoList.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"
#include "arcane/accelerator/core/internal/RunnerImpl.h"

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
    case eExecutionPolicy::SYCL:
      return impl::getSYCLRunQueueRuntime();
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

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerImpl::
initialize(Runner* runner, eExecutionPolicy v, DeviceId device)
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
  m_is_auto_prefetch_command = false;

  // Pour test
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_PREFETCH_COMMAND", true))
    m_is_auto_prefetch_command = (v.value() != 0);

  // Il faut initialiser le pool à la fin car il a besoin d'accéder à \a m_runtime
  m_run_queue_pool.initialize(runner->_impl());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerImpl::
_checkIsInit() const
{
  if (!m_is_init)
    ARCANE_FATAL("Runner is not initialized. Call method initialize() before");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerImpl::
_freePool()
{
  RunQueueImplStack& s = m_run_queue_pool;
  while (!s.empty()) {
    delete s.top();
    s.pop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImplStack* RunnerImpl::
getPool()
{
  return &m_run_queue_pool;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerImpl::
_internalPutRunQueueImplInPool(RunQueueImpl* p)
{
  RunnerImpl::Lock my_lock(this);
  getPool()->push(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunQueueImpl* RunnerImpl::
_internalCreateOrGetRunQueueImpl()
{
  _checkIsInit();

  auto pool = getPool();

  {
    impl::RunnerImpl::Lock my_lock(this);
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

impl::RunQueueImpl* RunnerImpl::
_internalCreateOrGetRunQueueImpl(const RunQueueBuildInfo& bi)
{
  _checkIsInit();
  // Si on utilise les paramètres par défaut, on peut utilier une RunQueueImpl
  // issue du pool.
  if (bi.isDefault())
    return _internalCreateOrGetRunQueueImpl();
  impl::IRunnerRuntime* runtime = m_runtime;
  ARCANE_CHECK_POINTER(runtime);
  auto* queue = new impl::RunQueueImpl(this, 0, bi);
  return queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImplStack::
createRunQueue(const RunQueueBuildInfo& bi)
{
  if (!m_runner_impl)
    ARCANE_FATAL("RunQueueImplStack is not initialized");
  Int32 x = ++m_nb_created;
  auto* q = new impl::RunQueueImpl(m_runner_impl, x, bi);
  q->m_is_in_pool = true;
  return q;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IRunQueueEventImpl* RunnerImpl::
_createEvent()
{
  _checkIsInit();
  return m_runtime->createEventImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IRunQueueEventImpl* RunnerImpl::
_createEventWithTimer()
{
  _checkIsInit();
  return m_runtime->createEventImplWithTimer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
Runner()
: m_p(std::make_shared<impl::RunnerImpl>())
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
  _checkIsInit();
  return m_p->runtime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eExecutionPolicy Runner::
executionPolicy() const
{
  return m_p->executionPolicy();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Runner::
isInitialized() const
{
  return m_p->isInit();
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

bool Runner::
_isAutoPrefetchCommand() const
{
  return m_p->isAutoPrefetchCommand();
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
arcaneCheckPointerIsAccessible(const Runner* runner, const void* ptr,
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
