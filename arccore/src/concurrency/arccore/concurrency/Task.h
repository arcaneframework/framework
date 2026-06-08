// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Task.h                                                      (C) 2000-2025 */
/*                                                                           */
/* Classes managing concurrent tasks.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_TASK_H
#define ARCCORE_BASE_TASK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/RangeFunctor.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ForLoopTraceInfo.h"
#include "arccore/base/ParallelLoopOptions.h"
#include "arccore/base/ForLoopRunInfo.h"

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO:
 * - Check memory leaks
 * - CLEARLY INDICATE THAT A TASK MUST NOT BE USED AFTER THE WAIT!!!
 * - Look into exception mechanism.
 * - Overload For and Foreach without specifying the block_size
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution context of a task.
 * \ingroup Concurrency
 */
class ARCCORE_CONCURRENCY_EXPORT TaskContext
{
 public:

  explicit TaskContext(ITask* atask)
  : m_task(atask)
  {}

 public:

  //! Current task.
  ITask* task() const { return m_task; }

 private:

  ITask* m_task;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a task functor.
 * \ingroup Concurrency
 */
class ARCCORE_CONCURRENCY_EXPORT ITaskFunctor
{
 public:

  virtual ~ITaskFunctor() = default;

 protected:

  ITaskFunctor(const ITaskFunctor&) = default;
  ITaskFunctor() = default;

 public:

  //! Executes the associated method
  virtual void executeFunctor(const TaskContext& tc) = 0;
  virtual ITaskFunctor* clone(void* buffer, Integer size) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Functor without arguments for a task.
 * \ingroup Concurrency
 */
template <typename InstanceType>
class TaskFunctor
: public ITaskFunctor
{
 public:

  typedef void (InstanceType::*FunctorType)();

 public:

  TaskFunctor(InstanceType* instance, FunctorType func)
  : m_instance(instance)
  , m_function(func)
  {
  }
  TaskFunctor(const TaskFunctor& rhs) = default;
  TaskFunctor& operator=(const TaskFunctor& rhs) = delete;

 public:

  //! Executes the associated method
  void executeFunctor(const TaskContext& /*tc*/) override
  {
    (m_instance->*m_function)();
  }
  ITaskFunctor* clone(void* buffer, Integer size) override
  {
    if (sizeof(*this) > (size_t)size)
      ARCCORE_FATAL("INTERNAL: task functor buffer is too small");
    return new (buffer) TaskFunctor<InstanceType>(*this);
  }

 private:

  InstanceType* m_instance;
  FunctorType m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Functor for a task taking a TaskContext as an argument.
 * \ingroup Concurrency
 */
template <typename InstanceType>
class TaskFunctorWithContext
: public ITaskFunctor
{
 public:

  typedef void (InstanceType::*FunctorType)(const TaskContext& tc);

 public:

  TaskFunctorWithContext(InstanceType* instance, FunctorType func)
  : ITaskFunctor()
  , m_instance(instance)
  , m_function(func)
  {
  }

 public:

  //! Executes the associated method
  void executeFunctor(const TaskContext& tc) override
  {
    (m_instance->*m_function)(tc);
  }
  ITaskFunctor* clone(void* buffer, Integer size) override
  {
    if (sizeof(*this) > (size_t)size)
      ARCCORE_FATAL("INTERNAL: task functor buffer is too small");
    return new (buffer) TaskFunctorWithContext<InstanceType>(*this);
  }

 private:

  InstanceType* m_instance = nullptr;
  FunctorType m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Concurrency
 * \brief Interface for a concurrent task.
 *
 * Tasks are created via TaskFactory.
 */
class ARCCORE_CONCURRENCY_EXPORT ITask
{
  friend class TaskFactory;

 public:

  virtual ~ITask() = default;

 public:

  /*!
   * \brief Launches the task and blocks until it finishes.
   *
   * After calling this function, the task is destroyed and must not
   * be used again.
   */
  virtual void launchAndWait() = 0;
  /*!
   * \brief Launches the child tasks \a tasks and blocks
   * until they finish.
   */
  virtual void launchAndWait(ConstArrayView<ITask*> tasks) = 0;

 protected:

  virtual ITask* _createChildTask(ITaskFunctor* functor) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
