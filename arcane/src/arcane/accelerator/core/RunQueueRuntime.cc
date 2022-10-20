// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueRuntime.cc                                          (C) 2000-2022 */
/*                                                                           */
/* Implémentation d'un RunQueue pour une cible donnée.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/IRunQueueRuntime.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/Memory.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/MemoryView.h"

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

  HostRunQueueStream(IRunQueueRuntime* runtime)
  : m_runtime(runtime)
  {}

 public:

  void notifyBeginLaunchKernel(RunCommandImpl&) override { return m_runtime->notifyBeginLaunchKernel(); }
  void notifyEndLaunchKernel(RunCommandImpl&) override { return m_runtime->notifyEndLaunchKernel(); }
  void barrier() override { return m_runtime->barrier(); }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    std::memcpy(args.destination().span().data(), args.source().span().data(), args.source().size());
  }
  void prefetchMemory(const MemoryPrefetchArgs&) override {}
  void* _internalImpl() override { return nullptr; }

 private:

  IRunQueueRuntime* m_runtime;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT HostRunQueueEvent
: public IRunQueueEventImpl
{
 public:
  HostRunQueueEvent(bool has_timer) : m_has_timer(has_timer){}
 public:
  void recordQueue(IRunQueueStream*) override {}
  void wait() override {}
  void waitForEvent(IRunQueueStream*) override {}
  Int64 elapsedTime(IRunQueueEventImpl*) final { return 0; }
 private:
  bool m_has_timer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT SequentialRunQueueRuntime
: public IRunQueueRuntime
{
 public:

  ~SequentialRunQueueRuntime() final = default;

 public:

  void notifyBeginLaunchKernel() final {}
  void notifyEndLaunchKernel() final {}
  void barrier() final {}
  eExecutionPolicy executionPolicy() const final { return eExecutionPolicy::Sequential; }
  IRunQueueStream* createStream(const RunQueueBuildInfo&) final { return new HostRunQueueStream(this); }
  IRunQueueEventImpl* createEventImpl() final { return new HostRunQueueEvent(false); }
  IRunQueueEventImpl* createEventImplWithTimer() final { return new HostRunQueueEvent(true); }
  void setMemoryAdvice(MemoryView, eMemoryAdvice, DeviceId) final {}
  void unsetMemoryAdvice(MemoryView, eMemoryAdvice, DeviceId) final {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT ThreadRunQueueRuntime
: public IRunQueueRuntime
{
 public:

  ~ThreadRunQueueRuntime() final = default;

 public:

  void notifyBeginLaunchKernel() final {}
  void notifyEndLaunchKernel() final {}
  void barrier() final {}
  eExecutionPolicy executionPolicy() const final { return eExecutionPolicy::Thread; }
  IRunQueueStream* createStream(const RunQueueBuildInfo&) final { return new HostRunQueueStream(this); }
  IRunQueueEventImpl* createEventImpl() final { return new HostRunQueueEvent(false); }
  IRunQueueEventImpl* createEventImplWithTimer() final { return new HostRunQueueEvent(true); }
  void setMemoryAdvice(MemoryView, eMemoryAdvice, DeviceId) final {}
  void unsetMemoryAdvice(MemoryView, eMemoryAdvice, DeviceId) final {}
};

namespace
{
SequentialRunQueueRuntime global_sequential_runqueue_runtime;
ThreadRunQueueRuntime global_thread_runqueue_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunQueueRuntime*
getSequentialRunQueueRuntime()
{
  return &global_sequential_runqueue_runtime;
}

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunQueueRuntime*
getThreadRunQueueRuntime()
{
  return &global_thread_runqueue_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
