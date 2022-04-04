﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyUtils.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Classes gérant la concurrence (tâches, boucles parallèles, ...)           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ConcurrencyUtils.h"

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Observable.h"

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

    m_functor = f->clone(functor_buf,FUNCTOR_CLASS_SIZE);
  }
 public:
  void launchAndWait() override
  {
    if (m_functor){
      ITaskFunctor* tmp_f = m_functor;
      m_functor = nullptr;
      TaskContext task_context(this);
      tmp_f->executeFunctor(task_context);
      delete this;
    }
  }
  void launchAndWait(ConstArrayView<ITask*> tasks) override
  {
    for( Integer i=0,n=tasks.size(); i<n; ++i )
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
  void executeParallelFor(Integer begin,Integer size,[[maybe_unused]] Integer block_size,IRangeFunctor* f) override
  {
    f->executeFunctor(begin,size);
  }
  void executeParallelFor(Integer begin,Integer size,[[maybe_unused]] const ParallelLoopOptions& options,IRangeFunctor* f) override
  {
    f->executeFunctor(begin,size);
  }
  void executeParallelFor(Integer begin,Integer size,IRangeFunctor* f) override
  {
    f->executeFunctor(begin,size);
  }
  void executeParallelFor(const ComplexLoopRanges<1>& loop_ranges,
                          [[maybe_unused]] const ParallelLoopOptions& options,
                          IMDRangeFunctor<1>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  void executeParallelFor(const ComplexLoopRanges<2>& loop_ranges,
                          [[maybe_unused]] const ParallelLoopOptions& options,
                          IMDRangeFunctor<2>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  void executeParallelFor(const ComplexLoopRanges<3>& loop_ranges,
                          [[maybe_unused]] const ParallelLoopOptions& options,
                          IMDRangeFunctor<3>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  void executeParallelFor(const ComplexLoopRanges<4>& loop_ranges,
                          [[maybe_unused]] const ParallelLoopOptions& options,
                          IMDRangeFunctor<4>* functor) override
  {
    functor->executeFunctor(loop_ranges);
  }
  bool isActive() const override
  {
    return false;
  }
  Int32 nbAllowedThread() const override
  {
    return 1;
  }
  Int32 currentTaskThreadIndex() const override
  {
    return 0;
  }
  Int32 currentTaskIndex() const override
  {
    return 0;
  }
  void setDefaultParallelLoopOptions(const ParallelLoopOptions& v) override
  {
    m_default_loop_options = v;
  }

  const ParallelLoopOptions& defaultParallelLoopOptions() override
  {
    return m_default_loop_options;
  }

  void printInfos(std::ostream& o) const final
  {
    o << "NullTaskImplementation";
 }

 private:
  ParallelLoopOptions m_default_loop_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NullTaskImplementation NullTaskImplementation::singleton;
ITaskImplementation* TaskFactory::m_impl = &NullTaskImplementation::singleton;
IObservable* TaskFactory::m_created_thread_observable = 0;
IObservable* TaskFactory::m_destroyed_thread_observable = 0;
Integer TaskFactory::m_verbose_level = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactory::
setImplementation(ITaskImplementation* task_impl)
{
  if (m_impl && m_impl!=&NullTaskImplementation::singleton)
    ARCANE_FATAL("TaskFactory already has an implementation");
  m_impl = task_impl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable*  TaskFactory::
createThreadObservable()
{
  if (!m_created_thread_observable)
    m_created_thread_observable = new Observable();
  return m_created_thread_observable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable*  TaskFactory::
destroyThreadObservable()
{
  if (!m_destroyed_thread_observable)
    m_destroyed_thread_observable = new Observable();
  return m_destroyed_thread_observable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactory::
terminate()
{
  // C'est celui qui a positionné l'implémentation qui gère sa destruction.
  if (m_impl==&NullTaskImplementation::singleton)
    return;
  if (m_impl)
    m_impl->terminate();
  m_impl = &NullTaskImplementation::singleton;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ConcurrencyUtils.h
 
 \brief Classes, Types et macros pour gérer la concurrence.

 Pour plus de renseignements, se reporter à la page \ref arcanedoc_concurrency
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
