// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueEvent.h                                             (C) 2000-2025 */
/*                                                                           */
/* Event on a run queue.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNQUEUEEVENT_H
#define ARCCORE_COMMON_ACCELERATOR_RUNQUEUEEVENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Ref.h"
#include "arccore/base/AutoRef2.h"

#include "arccore/common/accelerator/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Event for a run queue.
 *
 * This class has a reference semantics.
 *
 * The RunQueue::recordEvent() and RunQueue::waitEvent() methods allow
 * associating a RunQueueEvent with a given RunQueue to perform
 * a synchronization.
 *
 * For example:
 *
 * \snippet RunQueueUnitTest.cc SampleRunQueueEventSample1
 */
class ARCCORE_COMMON_EXPORT RunQueueEvent
{
  friend RunQueueEvent makeEvent(const Runner& runner);
  friend Ref<RunQueueEvent> makeEventRef(const Runner& runner);
  friend RunQueue;
  friend Impl::RunQueueImpl;
  class InternalImpl;

 private:

  //! Constructs an event. Use makeEvent() to construct an instance
  explicit RunQueueEvent(const Runner& runner);

 public:

  RunQueueEvent();
  RunQueueEvent(const RunQueueEvent&);
  RunQueueEvent& operator=(const RunQueueEvent&);
  RunQueueEvent(RunQueueEvent&&) noexcept;
  RunQueueEvent& operator=(RunQueueEvent&&) noexcept;
  ~RunQueueEvent();

 public:

  /*!
   * \brief Indicates if the instance is null.
   *
   * The instance is null if it was constructed with the default constructor.
   */
  bool isNull() const { return m_p.get() == nullptr; }

  //! Blocks until the queues associated with this event have finished
  //! their work.
  void wait();

  /*!
   * \brief Indicates if the RunQueues associated with this event have
   * finished their work.
   *
   * Returns \a false if the RunQueues registered via RunQueue::recordEvent() have
   * finished their work. Returns \a true otherwise.
   */
  bool hasPendingWork() const;

 private:

  Impl::IRunQueueEventImpl* _internalEventImpl() const;

 private:

  AutoRef2<InternalImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates an event associated with \a runner.
 */
inline RunQueueEvent
makeEvent(const Runner& runner)
{
  return RunQueueEvent(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates an event associated with \a runner.
 */
inline Ref<RunQueueEvent>
makeEventRef(const Runner& runner)
{
  return makeRef(new RunQueueEvent(runner));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
