// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITaskImplementation.h                                       (C) 2000-2025 */
/*                                                                           */
/* Task management interface.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ITASKIMPLEMENTATION_H
#define ARCCORE_BASE_ITASKIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Implementation of a task factory.
 *
 * \ingroup Concurrency
 *
 * This class is internal to Arcane. To manage tasks, you
 * must use the TaskFactory class.
 */
class ARCCORE_CONCURRENCY_EXPORT ITaskImplementation
{
 public:

  virtual ~ITaskImplementation() = default;

 public:

  /*!
   * \internal.
   * Initializes the implementation with a maximum of \a nb_thread.
   * If \a nb_thread is 0, the implementation can choose
   * the number of threads automatically.
   * This method is internal to Arcane and should only be called
   * during the execution initialization.
   */
  virtual void initialize(Int32 nb_thread) = 0;
  /*!
   * \internal.
   * Terminates the use of the implementation.
   * This method must be called only at the end of the calculation.
   */
  virtual void terminate() = 0;
  /*!
   * \brief Creates a root task.
   * The implementation must copy the value of \a f, which is either
   * a TaskFunctor or a TaskFunctorWithContext.
   */
  virtual ITask* createRootTask(ITaskFunctor* f) = 0;

  //! Executes the functor \a f in parallel.
  virtual void executeParallelFor(Integer begin, Integer size, const ParallelLoopOptions& options, IRangeFunctor* f) = 0;

  //! Executes the functor \a f in parallel.
  virtual void executeParallelFor(Integer begin, Integer size, Integer block_size, IRangeFunctor* f) = 0;

  //! Executes the functor \a f in parallel.
  virtual void executeParallelFor(Integer begin, Integer size, IRangeFunctor* f) = 0;

  //! Executes the loop \a loop_info in parallel.
  virtual void executeParallelFor(const ParallelFor1DLoopInfo& loop_info) = 0;

  //! Executes a 1D loop in parallel
  virtual void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<1>* functor) = 0;
  //! Executes a 2D loop in parallel
  virtual void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<2>* functor) = 0;
  //! Executes a 3D loop in parallel
  virtual void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<3>* functor) = 0;
  //! Executes a 4D loop in parallel
  virtual void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<4>* functor) = 0;

  //! Indicates if the implementation is active.
  virtual bool isActive() const = 0;

  //! Maximum number of threads used to manage tasks
  ARCCORE_DEPRECATED_REASON("Y2025: use ConcurrencyBase::maxAllowedThread() instead")
  Int32 nbAllowedThread() const;

  //! Implementation of TaskFactory::currentTaskThreadIndex()
  virtual Int32 currentTaskThreadIndex() const = 0;

  //! Implementation of TaskFactory::currentTaskIndex()
  virtual Int32 currentTaskIndex() const = 0;

  //! Prints information about the runtime used
  virtual void printInfos(std::ostream& o) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
