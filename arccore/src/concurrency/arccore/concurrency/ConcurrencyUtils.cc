// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyUtils.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Classes gérant la concurrence (tâches, boucles parallèles, ...)           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ITaskImplementation.h"

#include "arccore/concurrency/Task.h"
#include "arccore/concurrency/ParallelFor.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"

#include "arccore/base/Observable.h"

#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SerialTask
: public ITask
{
 public:

  typedef TaskFunctor<SerialTask> TaskType;

 public:

  static const int FUNCTOR_CLASS_SIZE = sizeof(TaskType);

 public:

  SerialTask(ITaskFunctor* f)
  : m_functor(f)
  {
    // \a f doit être une instance de TaskFunctor<SerialTask>.
    // on recopie dans un buffer pré-dimensionné pour éviter
    // d'avoir à faire une allocation sur le tas via le new
    // classique. On utilise donc le new avec placement.

    m_functor = f->clone(functor_buf, FUNCTOR_CLASS_SIZE);
  }

 public:

  void launchAndWait() override
  {
    if (m_functor) {
      ITaskFunctor* tmp_f = m_functor;
      m_functor = nullptr;
      TaskContext task_context(this);
      tmp_f->executeFunctor(task_context);
      delete this;
    }
  }
  void launchAndWait(ConstArrayView<ITask*> tasks) override
  {
    for (Integer i = 0, n = tasks.size(); i < n; ++i)
      tasks[i]->launchAndWait();
  }
  ITask* _createChildTask(ITaskFunctor* functor) override
  {
    return new SerialTask(functor);
  }

 private:

  ITaskFunctor* m_functor;
  char functor_buf[FUNCTOR_CLASS_SIZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullTaskImplementation
: public ITaskImplementation
{
 public:

  static NullTaskImplementation singleton;

 public:

  void initialize([[maybe_unused]] Int32 nb_thread) override
  {
  }
  void terminate() override
  {
  }
  ITask* createRootTask(ITaskFunctor* f) override
  {
    return new SerialTask(f);
  }
  void executeParallelFor(Integer begin, Integer size,
                          [[maybe_unused]] Integer block_size, IRangeFunctor* f) override
  {
    f->executeFunctor(begin, size);
  }
  void executeParallelFor(Integer begin, Integer size,
                          [[maybe_unused]] const ParallelLoopOptions& options,
                          IRangeFunctor* f) override
  {
    f->executeFunctor(begin, size);
  }
  void executeParallelFor(Integer begin, Integer size, IRangeFunctor* f) override
  {
    f->executeFunctor(begin, size);
  }
  void executeParallelFor(const ParallelFor1DLoopInfo& loop_info) override
  {
    loop_info.functor()->executeFunctor(loop_info.beginIndex(), loop_info.size());
  }
  void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                          [[maybe_unused]] const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<1>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                          [[maybe_unused]] const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<2>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                          [[maybe_unused]] const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<3>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                          [[maybe_unused]] const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<4>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  bool isActive() const override
  {
    return false;
  }
  Int32 currentTaskThreadIndex() const override
  {
    return 0;
  }
  Int32 currentTaskIndex() const override
  {
    return 0;
  }

  void printInfos(std::ostream& o) const final
  {
    o << "NullTaskImplementation";
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NullTaskImplementation NullTaskImplementation::singleton;
ITaskImplementation* TaskFactory::m_impl = &NullTaskImplementation::singleton;
Int32 TaskFactory::m_verbose_level = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  IObservable* global_created_thread_observable = 0;
  IObservable* global_destroyed_thread_observable = 0;
  std::mutex global_observable_mutex;

  IObservable*
  _checkCreateGlobalThreadObservable()
  {
    if (!global_created_thread_observable)
      global_created_thread_observable = new Observable();
    return global_created_thread_observable;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactoryInternal::
setImplementation(ITaskImplementation* task_impl)
{
  if (TaskFactory::m_impl && TaskFactory::m_impl != &NullTaskImplementation::singleton)
    ARCCORE_FATAL("TaskFactory already has an implementation");
  TaskFactory::m_impl = task_impl;
}

void TaskFactoryInternal::
addThreadCreateObserver(IObserver* o)
{
  std::scoped_lock slock(global_observable_mutex);
  _checkCreateGlobalThreadObservable();
  global_created_thread_observable->attachObserver(o);
}

void TaskFactoryInternal::
removeThreadCreateObserver(IObserver* o)
{
  std::scoped_lock slock(global_observable_mutex);
  _checkCreateGlobalThreadObservable();
  global_created_thread_observable->detachObserver(o);
}

void TaskFactoryInternal::
notifyThreadCreated()
{
  std::scoped_lock slock(global_observable_mutex);
  if (global_created_thread_observable)
    global_created_thread_observable->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactory::
_internalSetImplementation(ITaskImplementation* task_impl)
{
  TaskFactoryInternal::setImplementation(task_impl);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable* TaskFactory::
createThreadObservable()
{
  std::scoped_lock slock(global_observable_mutex);
  return _checkCreateGlobalThreadObservable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable* TaskFactory::
destroyThreadObservable()
{
  if (!global_destroyed_thread_observable)
    global_destroyed_thread_observable = new Observable();
  return global_destroyed_thread_observable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactory::
terminate()
{
  // C'est celui qui a positionné l'implémentation qui gère sa destruction.
  if (m_impl == &NullTaskImplementation::singleton)
    return;
  if (m_impl)
    m_impl->terminate();
  m_impl = &NullTaskImplementation::singleton;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ConcurrencyUtils.h
 * 
 * \brief Classes, Types et macros pour gérer la concurrence.
 *
 * Pour plus de renseignements, se reporter à la page \ref arcanedoc_parallel_concurrency
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
