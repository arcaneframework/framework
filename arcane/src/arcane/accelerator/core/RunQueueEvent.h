// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueEvent.h                                             (C) 2000-2025 */
/*                                                                           */
/* Evènement sur une file d'exécution.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNQUEUEEVENT_H
#define ARCANE_ACCELERATOR_CORE_RUNQUEUEEVENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/utils/AutoRef.h"

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
  friend RunQueueEvent makeEvent(const Runner& runner);
  friend Ref<RunQueueEvent> makeEventRef(const Runner& runner);
  friend RunQueue;
  friend impl::RunQueueImpl;
  class Impl;

 private:

  //! Construit un évènement. Utiliser makeEvent() pour constuire une instance
  explicit RunQueueEvent(const Runner& runner);

 public:

  ~RunQueueEvent();
  RunQueueEvent(const RunQueueEvent&) = delete;
  RunQueueEvent& operator=(const RunQueueEvent&) = delete;

 public:

  void wait();

 private:

  impl::IRunQueueEventImpl* _internalEventImpl() const;

 private:

  AutoRef2<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un évènement associé à \a runner.
 */
inline RunQueueEvent
makeEvent(const Runner& runner)
{
  return RunQueueEvent(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un évènement associé à \a runner.
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
