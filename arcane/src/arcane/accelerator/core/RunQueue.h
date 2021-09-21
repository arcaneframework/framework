// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.h                                                  (C) 2000-2021 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNQUEUE_H
#define ARCANE_ACCELERATOR_CORE_RUNQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
class RunQueueImpl;
class RunCommandImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief File d'exécution  pour accélérateur.
 * \warning API en cours de définition.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunQueue
{
  friend class RunCommand;
 public:
  RunQueue(Runner& runner);
  RunQueue(Runner& runner,eExecutionPolicy policy);
  ~RunQueue();
  RunQueue(const RunQueue&) = delete;
  RunQueue& operator=(const RunQueue&) = delete;
 public:
  eExecutionPolicy executionPolicy() const;
  void setAsync(bool v) { m_is_async = v; }
  bool isAsync() const { return m_is_async; }
  void barrier();
 public:
  IRunQueueRuntime* _internalRuntime() const;
  IRunQueueStream* _internalStream() const;
 private:
  RunCommandImpl* _getCommandImpl();
 private:
  RunQueueImpl* m_p;
  bool m_is_async = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une commande associée à la file \a run_queue.
 */
inline RunCommand
makeCommand(RunQueue& run_queue)
{
  return RunCommand(run_queue);
}

/*!
 * \brief Créé une commande associée à la file \a run_queue.
 */
inline RunCommand
makeCommand(RunQueue* run_queue)
{
  return RunCommand(*run_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
