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

  void notifyBeginKernel(RunCommand&) override { return m_runtime->notifyBeginKernel(); }
  void notifyEndKernel(RunCommand&) override { return m_runtime->notifyEndKernel(); }
  void barrier() override { return m_runtime->barrier(); }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    std::memcpy(args.destination().data(), args.source().data(), args.source().size());
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
  HostRunQueueEvent(){}
 public:
  void recordQueue(IRunQueueStream*) override {}
  void wait() override {}
  void waitForEvent(IRunQueueStream*) override {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT SequentialRunQueueRuntime
: public IRunQueueRuntime
{
 public:
  ~SequentialRunQueueRuntime() override = default;
 public:
  void notifyBeginKernel() override {}
  void notifyEndKernel() override {}
  void barrier() override {}
  eExecutionPolicy executionPolicy() const override { return eExecutionPolicy::Sequential; }
  IRunQueueStream* createStream(const RunQueueBuildInfo&) override { return new HostRunQueueStream(this); }
  IRunQueueEventImpl* createEventImpl() override { return new HostRunQueueEvent(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT ThreadRunQueueRuntime
: public IRunQueueRuntime
{
 public:
  ~ThreadRunQueueRuntime() override = default;
 public:
  void notifyBeginKernel() override {}
  void notifyEndKernel() override {}
  void barrier() override {}
  eExecutionPolicy executionPolicy() const override { return eExecutionPolicy::Thread; }
  IRunQueueStream* createStream(const RunQueueBuildInfo&) override { return new HostRunQueueStream(this); }
  IRunQueueEventImpl* createEventImpl() override { return new HostRunQueueEvent(); }
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
