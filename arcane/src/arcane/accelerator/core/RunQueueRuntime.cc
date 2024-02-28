﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueRuntime.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Implémentation d'un RunQueue pour une cible donnée.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/DeviceInfoList.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT HostRunQueueStream
: public IRunQueueStream
{
 public:

  HostRunQueueStream(IRunnerRuntime* runtime)
  : m_runtime(runtime)
  {}

 public:

  void notifyBeginLaunchKernel(RunCommandImpl&) override { return m_runtime->notifyBeginLaunchKernel(); }
  void notifyEndLaunchKernel(RunCommandImpl&) override { return m_runtime->notifyEndLaunchKernel(); }
  void barrier() override { return m_runtime->barrier(); }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    args.destination().copyHost(args.source());
  }
  void prefetchMemory(const MemoryPrefetchArgs&) override {}
  void* _internalImpl() override { return nullptr; }
  bool _barrierNoException() override { return false; }

 private:

  IRunnerRuntime* m_runtime;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT HostRunQueueEvent
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
  Int64 elapsedTime(IRunQueueEventImpl* start_event) final
  {
    ARCANE_CHECK_POINTER(start_event);
    auto* true_start_event = static_cast<HostRunQueueEvent*>(start_event);
    if (!m_has_timer || !true_start_event->m_has_timer)
      ARCANE_FATAL("Event has no timer support");
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

class ARCANE_ACCELERATOR_CORE_EXPORT CommonRunnerRuntime
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
    _fillPointerAttribute(attribute,ptr);
  }

 private:

  DeviceInfoList m_device_info_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT SequentialRunnerRuntime
: public CommonRunnerRuntime
{
  eExecutionPolicy executionPolicy() const final { return eExecutionPolicy::Sequential; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT ThreadRunnerRuntime
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
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getSequentialRunQueueRuntime()
{
  return &global_sequential_runqueue_runtime;
}

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getThreadRunQueueRuntime()
{
  return &global_thread_runqueue_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
