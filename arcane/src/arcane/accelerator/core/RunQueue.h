// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.h                                                  (C) 2000-2022 */
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

  explicit RunQueue(Runner& runner);
  RunQueue(Runner& runner, const RunQueueBuildInfo& bi);
  RunQueue(Runner& runner, eExecutionPolicy policy);
  ~RunQueue();
  RunQueue(const RunQueue&) = delete;
  RunQueue& operator=(const RunQueue&) = delete;

 public:

  eExecutionPolicy executionPolicy() const;
  /*!
   * \brief Positionne l'asynchronisme de l'instance.
   *
   * Si l'instance est asynchrone, il faut appeler explicitement barrier()
   * pour attendre la fin de l'exécution des commandes.
   */
  void setAsync(bool v) { m_is_async = v; }
  //! Indique si la file d'exécution est asynchrone.
  bool isAsync() const { return m_is_async; }
  //! Bloque tant que toutes les commandes associées à la file ne sont pas terminées.
  void barrier();

  //! Copie des informations entre deux zones mémoires
  void copyMemory(const MemoryCopyArgs& args);

 public:

  impl::IRunQueueRuntime* _internalRuntime() const;
  impl::IRunQueueStream* _internalStream() const;

 private:

  impl::RunCommandImpl* _getCommandImpl();

 private:

  impl::RunQueueImpl* m_p;
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
  ARCANE_CHECK_POINTER(run_queue);
  return RunCommand(*run_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
