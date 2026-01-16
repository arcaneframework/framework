// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.cc                                                   (C) 2000-2025 */
/*                                                                           */
/* Gestion de l'exécution sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/Runner.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/MemoryView.h"
#include "arccore/base/Profiling.h"
#include "arccore/base/Convert.h"
#include "arccore/base/internal/ProfilingInternal.h"

#include "arccore/common/accelerator/RunQueueBuildInfo.h"
#include "arccore/common/accelerator/DeviceId.h"
#include "arccore/common/accelerator/DeviceMemoryInfo.h"
#include "arccore/common/accelerator/IDeviceInfoList.h"
#include "arccore/common/accelerator/PointerAttribute.h"
#include "arccore/common/accelerator/KernelLaunchArgs.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"
#include "arccore/common/accelerator/internal/RunnerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

namespace
{
  inline Impl::IRunnerRuntime*
  _getRuntimeNoCheck(eExecutionPolicy p)
  {
    Impl::IRunnerRuntime* runtime = nullptr;
    switch (p) {
    case eExecutionPolicy::None:
      ARCCORE_FATAL("No runtime for eExecutionPolicy::None");
    case eExecutionPolicy::SYCL:
      return Impl::getSYCLRunQueueRuntime();
    case eExecutionPolicy::HIP:
      return Impl::getHIPRunQueueRuntime();
    case eExecutionPolicy::CUDA:
      return Impl::getCUDARunQueueRuntime();
    case eExecutionPolicy::Sequential:
      return Impl::getSequentialRunQueueRuntime();
    case eExecutionPolicy::Thread:
      return Impl::getThreadRunQueueRuntime();
    }
    return runtime;
  }

  inline Impl::IRunnerRuntime*
  _getRuntime(eExecutionPolicy p)
  {
    auto* x = _getRuntimeNoCheck(p);
    if (!x)
      ARCCORE_FATAL("No runtime is available for execution policy '{0}'", p);
    return x;
  }

  inline void
  _stopProfiling(eExecutionPolicy p)
  {
    auto* x = _getRuntimeNoCheck(p);
    if (x)
      x->stopProfiling();
  }
  inline void
  _finalize(eExecutionPolicy p, ITraceMng* tm)
  {
    auto* x = _getRuntimeNoCheck(p);
    if (x) {
      x->stopProfiling();
      x->finalize(tm);
    }
  }
} // namespace

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerImpl::
initialize(Runner* runner, eExecutionPolicy v, DeviceId device)
{
  if (m_is_init)
    ARCCORE_FATAL("Runner is already initialized");
  if (v == eExecutionPolicy::None)
    ARCCORE_THROW(ArgumentException, "executionPolicy should not be eExecutionPolicy::None");
  if (device.isHost() || device.isNull())
    ARCCORE_THROW(ArgumentException, "device should not be Device::hostDevice() or Device::nullDevice()");

  m_execution_policy = v;
  m_device_id = device;
  m_runtime = _getRuntime(v);
  m_device_info = m_runtime->deviceInfoList()->deviceInfo(m_device_id.asInt32());
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
    ARCCORE_FATAL("Runner is not initialized. Call method initialize() before");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerImpl::
_freePool()
{
  RunQueueImplStack& s = m_run_queue_pool;
  while (!s.empty()) {
    RunQueueImpl* q = s.top();
    RunQueueImpl::_destroy(q);
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

Impl::RunQueueImpl* RunnerImpl::
_internalCreateOrGetRunQueueImpl()
{
  _checkIsInit();

  auto pool = getPool();

  {
    Impl::RunnerImpl::Lock my_lock(this);
    if (!pool->empty()) {
      Impl::RunQueueImpl* p = pool->top();
      pool->pop();
      return p;
    }
  }

  return pool->createRunQueue(RunQueueBuildInfo{});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::RunQueueImpl* RunnerImpl::
_internalCreateOrGetRunQueueImpl(const RunQueueBuildInfo& bi)
{
  _checkIsInit();
  // Si on utilise les paramètres par défaut, on peut utilier une RunQueueImpl
  // issue du pool.
  if (bi.isDefault())
    return _internalCreateOrGetRunQueueImpl();
  Impl::IRunnerRuntime* runtime = m_runtime;
  ARCCORE_CHECK_POINTER(runtime);
  auto* queue = new Impl::RunQueueImpl(this, 0, bi);
  return queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImplStack::
createRunQueue(const RunQueueBuildInfo& bi)
{
  if (!m_runner_impl)
    ARCCORE_FATAL("RunQueueImplStack is not initialized");
  Int32 x = ++m_nb_created;
  auto* q = new Impl::RunQueueImpl(m_runner_impl, x, bi);
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

} // namespace Arcane::Accelerator::Impl

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
: m_p(std::make_shared<Impl::RunnerImpl>())
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

Impl::IRunnerRuntime* Runner::
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
setConcurrentQueueCreation(bool)
{
  // Toujours thread-safe;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Runner::
isConcurrentQueueCreation() const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
setDeviceReducePolicy(eDeviceReducePolicy v)
{
  if (v != eDeviceReducePolicy::Grid)
    std::cout << "Warning: Runner::setDeviceReducePolicy(): only 'eDeviceReducePolicy::Grid' is supported\n";
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
    ARCCORE_FATAL("Runner is not initialized. Call method initialize() before");
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
    ARCCORE_FATAL("Internal error: no device available");
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
  Impl::IRunnerRuntime* r = _getRuntime(policy);
  return r->deviceInfoList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DeviceMemoryInfo Runner::
deviceMemoryInfo() const
{
  _checkIsInit();
  return m_p->runtime()->getDeviceMemoryInfo(deviceId());
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

bool Runner::
_isAutoPrefetchCommand() const
{
  return m_p->isAutoPrefetchCommand();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunnerInternal* Runner::
_internalApi()
{
  return m_p->_internalApi();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arrête tout les profiling en cours de tout les runtime.
 *
 * En général on utilise cela en fin de calcul.
 */
void RunnerInternal::
stopAllProfiling()
{
  _stopProfiling(eExecutionPolicy::CUDA);
  _stopProfiling(eExecutionPolicy::HIP);
  _stopProfiling(eExecutionPolicy::Sequential);
  _stopProfiling(eExecutionPolicy::Thread);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerInternal::
finalize(ITraceMng* tm)
{
  _finalize(eExecutionPolicy::CUDA, tm);
  _finalize(eExecutionPolicy::HIP, tm);
  _finalize(eExecutionPolicy::Sequential, tm);
  _finalize(eExecutionPolicy::Thread, tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerInternal::
startProfiling()
{
  m_runner_impl->runtime()->startProfiling();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerInternal::
stopProfiling()
{
  m_runner_impl->runtime()->stopProfiling();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunnerInternal::
isProfilingActive()
{
  return m_runner_impl->runtime()->isProfilingActive();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunnerInternal::
printProfilingInfos(std::ostream& o)
{
  bool is_active = isProfilingActive();
  if (is_active)
    stopProfiling();

  {
    // Affiche les statistiques de profiling.
    using Arcane::Impl::AcceleratorStatInfoList;
    auto f = [&](const AcceleratorStatInfoList& stat_list) {
      stat_list.print(o);
    };
    ProfilingRegistry::visitAcceleratorStat(f);
  }

  if (is_active)
    startProfiling();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ePointerAccessibility
getPointerAccessibility(Runner* runner, const void* ptr, PointerAttribute* ptr_attr)
{
  if (!runner)
    return ePointerAccessibility::Unknown;
  return Impl::RuntimeStaticInfo::getPointerAccessibility(runner->executionPolicy(), ptr, ptr_attr);
}

extern "C++" void Impl::
arcaneCheckPointerIsAccessible(const Runner* runner, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  if (!runner)
    return;
  return Impl::RuntimeStaticInfo::checkPointerIsAcccessible(runner->executionPolicy(), ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::IRunnerRuntime::
_fillPointerAttribute(PointerAttribute& attribute,
                      ePointerMemoryType mem_type,
                      int device, const void* pointer, const void* device_pointer,
                      const void* host_pointer)
{
  attribute = PointerAttribute(mem_type, device, pointer, device_pointer, host_pointer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::IRunnerRuntime::
_fillPointerAttribute(PointerAttribute& attribute, const void* pointer)
{
  attribute = PointerAttribute(pointer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::KernelLaunchArgs Impl::IRunnerRuntime::
computeKernalLaunchArgs(const Impl::KernelLaunchArgs& orig_args,
                        [[maybe_unused]] const void* kernel_ptr,
                        [[maybe_unused]] Int64 total_loop_size)
{
  return orig_args;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
