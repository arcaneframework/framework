// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskFactory.h                                               (C) 2000-2025 */
/*                                                                           */
/* Factory for tasks.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_TASKFACTORY_H
#define ARCCORE_CONCURRENCY_TASKFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ConcurrencyBase.h"
#include "arccore/concurrency/Task.h"
#include "arccore/concurrency/ITaskImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Concurrency
 * \brief Factory for tasks.
 */
class ARCCORE_CONCURRENCY_EXPORT TaskFactory
{
  friend TaskFactoryInternal;

 public:

  TaskFactory() = delete;

 public:

  /*!
   * \brief Creates a task.
   * During execution, the task will call the method \a function via
   * the instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createTask(InstanceType* instance, void (InstanceType::*function)(const TaskContext& tc))
  {
    TaskFunctorWithContext<InstanceType> functor(instance, function);
    return m_impl->createRootTask(&functor);
  }

  /*!
   * \brief Creates a task.
   * During execution, the task will call the method \a function via
   * the instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createTask(InstanceType* instance, void (InstanceType::*function)())
  {
    TaskFunctor<InstanceType> functor(instance, function);
    return m_impl->createRootTask(&functor);
  }

  /*!
   * \brief Creates a child task.
   *
   * During execution, the task will call the method \a function via
   * the instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createChildTask(ITask* parent_task, InstanceType* instance, void (InstanceType::*function)(const TaskContext& tc))
  {
    ARCCORE_CHECK_POINTER(parent_task);
    TaskFunctorWithContext<InstanceType> functor(instance, function);
    return parent_task->_createChildTask(&functor);
  }

  /*!
   * \brief Creates a child task.
   *
   * During execution, the task will call the method \a function via
   * the instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createChildTask(ITask* parent_task, InstanceType* instance, void (InstanceType::*function)())
  {
    ARCCORE_CHECK_POINTER(parent_task);
    TaskFunctor<InstanceType> functor(instance, function);
    return parent_task->_createChildTask(&functor);
  }

  //! Executes the functor \a f in parallel.
  static void executeParallelFor(Integer begin, Integer size, const ParallelLoopOptions& options, IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin, size, options, f);
  }

  //! Executes the functor \a f in parallel.
  static void executeParallelFor(Integer begin, Integer size, Integer block_size, IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin, size, block_size, f);
  }

  //! Executes the functor \a f in parallel.
  static void executeParallelFor(Integer begin, Integer size, IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin, size, f);
  }

  //! Executes the loop \a loop_info in parallel.
  static void executeParallelFor(const ParallelFor1DLoopInfo& loop_info)
  {
    m_impl->executeParallelFor(loop_info);
  }

  //! Executes a simple loop
  static void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<1>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Executes a simple loop
  static void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<1>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Executes a 2D loop
  static void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<2>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Executes a 2D loop
  static void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<2>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Executes a 3D loop
  static void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<3>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Executes a 3D loop
  static void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<3>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Executes a 4D loop
  static void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<4>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Executes a 4D loop
  static void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<4>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Maximum number of threads used to manage tasks.
  static Int32 nbAllowedThread() { return ConcurrencyBase::maxAllowedThread(); }

  /*!
   * \brief Index (between 0 and nbAllowedThread()-1) of the thread executing
   * the current task.
   *
   * For performance reasons, it is preferable to call this method
   * as little as possible. Ideally, it should only be called at the beginning
   * of the task execution and then the returned value should be used.
   */
  static Int32 currentTaskThreadIndex()
  {
    return m_impl->currentTaskThreadIndex();
  }

  /*!
   * \brief Index (between 0 and nbAllowedThread()-1) of the current task.
   *
   * This index is the same as currentTaskThreadIndex() except when we are in
   * an executeParallelFor() with deterministic partitioning
   * (ParallelLoopOptions::Partitioner::Deterministic).
   * In the latter case, the task number is assigned deterministically, depending
   * only on the number of threads allocated for the task and
   * ParallelLoopOptions::grainSize().
   *
   * If the current thread is not executing a task associated with this implementation,
   * it returns (-1).
   */
  static Int32 currentTaskIndex()
  {
    return m_impl->currentTaskIndex();
  }

 public:

  // TODO: mark these two methods as obsolete and indicate using
  // those from ConcurrencyBase instead.

  //! Sets the default parallel loop execution options
  static void setDefaultParallelLoopOptions(const ParallelLoopOptions& v)
  {
    ConcurrencyBase::setDefaultParallelLoopOptions(v);
  }

  //! Default parallel loop execution options
  static const ParallelLoopOptions& defaultParallelLoopOptions()
  {
    return ConcurrencyBase::defaultParallelLoopOptions();
  }

 public:

  /*!
   * \brief Indicates whether tasks are active.
   * Tasks are active if an implementation is available and if the requested number
   * of threads is strictly greater than 1.
   */
  static bool isActive()
  {
    return m_impl->isActive();
  }

  /*!
   * \brief Prints information about the implementation.
   *
   * The information is for example the version number or the name
   * of the implementation.
   */
  static void printInfos(std::ostream& o)
  {
    return m_impl->printInfos(o);
  }

  /*!
   * \brief Observable called when a thread is created for a task.
   *
   * \warning The observable instance is created during the first call
   * to this method. It is therefore not thread-safe. Similarly,
   * modifying the observable (adding/removing observers)
   * is not thread-safe.
   */
  ARCCORE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Do not use it")
  static IObservable* createThreadObservable();

  /*!
   * \brief Observable called when a thread is destroyed for a task.
   *
   * \warning The observable instance is created during the first call
   * to this method. It is therefore not thread-safe. Similarly,
   * modifying the observable (adding/removing observers)
   * is not thread-safe.
   */
  ARCCORE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Do not use it")
  static IObservable* destroyThreadObservable();

  /*!
   * \internal
   * \brief Indicates that threads will no longer be used.
   * This method must not be called when tasks are active.
   */
  static void terminate();

 public:

  //! Sets the verbosity level (0 for no output, which is the default)
  static void setVerboseLevel(Integer v) { m_verbose_level = v; }

  //! Verbosity level
  static Integer verboseLevel() { return m_verbose_level; }

 public:

  //! \internal
  ARCCORE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. "
                            "Use TaskFactoryInternal::setImplementation() instead")
  static void _internalSetImplementation(ITaskImplementation* task_impl);

 private:

  static ITaskImplementation* m_impl;
  static Int32 m_verbose_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
