// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueRuntime.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Implémentation d'un RunQueue pour une cible donnée.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/IRunQueueRuntime.h"
#include "arcane/accelerator/IRunQueueStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_EXPORT HostRunQueueStream
: public IRunQueueStream
{
 public:
  HostRunQueueStream(IRunQueueRuntime* runtime): m_runtime(runtime){}
 public:
  void notifyBeginKernel() override { return m_runtime->notifyBeginKernel(); }
  void notifyEndKernel() override { return m_runtime->notifyEndKernel(); }
  void barrier() override { return m_runtime->barrier(); }
  void* _internalImpl() override { return nullptr; }
 private:
  IRunQueueRuntime* m_runtime;
};

class ARCANE_ACCELERATOR_EXPORT SequentialRunQueueRuntime
: public IRunQueueRuntime
{
 public:
  ~SequentialRunQueueRuntime() override = default;
 public:
  void notifyBeginKernel() override {}
  void notifyEndKernel() override {}
  void barrier() override {}
  eExecutionPolicy executionPolicy() const override { return eExecutionPolicy::Sequential; }
  IRunQueueStream* createStream() override { return new HostRunQueueStream(this); }
};

class ARCANE_ACCELERATOR_EXPORT ThreadRunQueueRuntime
: public IRunQueueRuntime
{
 public:
  ~ThreadRunQueueRuntime() override = default;
 public:
  void notifyBeginKernel() override {}
  void notifyEndKernel() override {}
  void barrier() override {}
  eExecutionPolicy executionPolicy() const override { return eExecutionPolicy::Thread; }
  IRunQueueStream* createStream() override { return new HostRunQueueStream(this); }
};

namespace
{
SequentialRunQueueRuntime global_sequential_runqueue_runtime;
ThreadRunQueueRuntime global_thread_runqueue_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_EXPORT
IRunQueueRuntime* getSequentialRunQueueRuntime()
{
  return &global_sequential_runqueue_runtime;
}

//! Récupère l'implémentation séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_EXPORT
IRunQueueRuntime* getThreadRunQueueRuntime()
{
  return &global_thread_runqueue_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
