// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueRuntime.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'un RunQueue pour une cible donnée.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/MemoryView.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/common/MemoryUtils.h"

#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/IRunQueueStream.h"
#include "arccore/common/accelerator/internal/IRunQueueEventImpl.h"
#include "arccore/common/accelerator/Memory.h"
#include "arccore/common/accelerator/DeviceInfoList.h"
#include "arccore/common/accelerator/DeviceMemoryInfo.h"
#include "arccore/common/accelerator/NativeStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT HostRunQueueStream
: public IRunQueueStream
{
 public:

  explicit HostRunQueueStream(IRunnerRuntime* runtime)
  : m_runtime(runtime)
  {}

 public:

  void notifyBeginLaunchKernel(RunCommandImpl&) override { return m_runtime->notifyBeginLaunchKernel(); }
  void notifyEndLaunchKernel(RunCommandImpl&) override { return m_runtime->notifyEndLaunchKernel(); }
  void barrier() override { return m_runtime->barrier(); }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    MemoryUtils::copyHost(args.destination(), args.source());
  }
  void prefetchMemory(const MemoryPrefetchArgs&) override {}
  Impl::NativeStream nativeStream() override { return {}; }
  bool _barrierNoException() override { return false; }

 private:

  IRunnerRuntime* m_runtime;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT HostRunQueueEvent
: public IRunQueueEventImpl
{
 public:

  explicit HostRunQueueEvent(bool has_timer)
  : m_has_timer(has_timer)
  {}

 public:

  void recordQueue(IRunQueueStream*) final
  {
    if (m_has_timer)
      m_recorded_time = platform::getRealTime();
  }
  void wait() final {}
  void waitForEvent(IRunQueueStream*) final {}
  bool hasPendingWork() final { return false; }
  Int64 elapsedTime(IRunQueueEventImpl* start_event) final
  {
    ARCCORE_CHECK_POINTER(start_event);
    auto* true_start_event = static_cast<HostRunQueueEvent*>(start_event);
    if (!m_has_timer || !true_start_event->m_has_timer)
      ARCCORE_FATAL("Event has no timer support");
    double diff_time = m_recorded_time - true_start_event->m_recorded_time;
    Int64 diff_as_int64 = static_cast<Int64>(diff_time * 1.0e9);
    return diff_as_int64;
  }

 private:

  bool m_has_timer = false;
  double m_recorded_time = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT CommonRunnerRuntime
: public IRunnerRuntime
{
 public:

  CommonRunnerRuntime()
  {
    DeviceInfo d;
    d.setDeviceId(DeviceId());
    d.setName("HostDevice");
    m_device_info_list.addDevice(d);
  }

 public:

  void notifyBeginLaunchKernel() final {}
  void notifyEndLaunchKernel() final {}
  void barrier() final {}
  IRunQueueStream* createStream(const RunQueueBuildInfo&) final { return new HostRunQueueStream(this); }
  IRunQueueEventImpl* createEventImpl() final { return new HostRunQueueEvent(false); }
  IRunQueueEventImpl* createEventImplWithTimer() final { return new HostRunQueueEvent(true); }
  void setMemoryAdvice(ConstMemoryView, eMemoryAdvice, DeviceId) final {}
  void unsetMemoryAdvice(ConstMemoryView, eMemoryAdvice, DeviceId) final {}
  void setCurrentDevice(DeviceId) final {}
  const IDeviceInfoList* deviceInfoList() final { return &m_device_info_list; }
  void getPointerAttribute(PointerAttribute& attribute, const void* ptr) final
  {
    _fillPointerAttribute(attribute, ptr);
  }
  DeviceMemoryInfo getDeviceMemoryInfo(DeviceId) override
  {
    // TODO: à implémenter
    return {};
  }

 private:

  DeviceInfoList m_device_info_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT SequentialRunnerRuntime
: public CommonRunnerRuntime
{
  eExecutionPolicy executionPolicy() const final { return eExecutionPolicy::Sequential; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT ThreadRunnerRuntime
: public CommonRunnerRuntime
{
 public:

  eExecutionPolicy executionPolicy() const final { return eExecutionPolicy::Thread; }
};

namespace
{
  SequentialRunnerRuntime global_sequential_runqueue_runtime;
  ThreadRunnerRuntime global_thread_runqueue_runtime;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getSequentialRunQueueRuntime()
{
  return &global_sequential_runqueue_runtime;
}

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getThreadRunQueueRuntime()
{
  return &global_thread_runqueue_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
