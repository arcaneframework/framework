// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueEvent.h                                             (C) 2000-2024 */
/*                                                                           */
/* Evènement sur une file d'exécution.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNQUEUEEVENT_H
#define ARCANE_ACCELERATOR_CORE_RUNQUEUEEVENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arcane/accelerator/core/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Evènement pour une file d'exécution.
 * \warning API en cours de définition.
 *
 * Les méthodes RunQueue::recordEvent() et RunQueue::waitEvent() permettent
 * d'associer un RunQueueEvent à une RunQueue donnée.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunQueueEvent
{
  friend RunQueueEvent makeEvent(Runner& runner);
  friend Ref<RunQueueEvent> makeEventRef(Runner& runner);
  friend RunQueue;
  friend impl::RunQueueImpl;

 private:

  //! Construit un évènement. Utiliser makeEvent() pour constuire une instance
  explicit RunQueueEvent(Runner& runner);

 public:

  ~RunQueueEvent();
  RunQueueEvent(const RunQueueEvent&) = delete;
  RunQueueEvent& operator=(const RunQueueEvent&) = delete;

 public:

  void wait();

 private:

  impl::IRunQueueEventImpl* _internalEventImpl() const { return m_p; }

 private:

  impl::IRunQueueEventImpl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un évènement associé à \a runner.
 */
inline RunQueueEvent
makeEvent(Runner& runner)
{
  return RunQueueEvent(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un évènement associé à \a runner.
 */
inline Ref<RunQueueEvent>
makeEventRef(Runner& runner)
{
  return makeRef(new RunQueueEvent(runner));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
