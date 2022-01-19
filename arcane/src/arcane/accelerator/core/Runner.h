// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.h                                                    (C) 2000-2021 */
/*                                                                           */
/* Gestion de l'exécution sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNNER_H
#define ARCANE_ACCELERATOR_CORE_RUNNER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
class RunQueueImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire d'exécution pour accélérateur.
 * \warning API en cours de définition.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT Runner
{
  friend class RunQueueImpl;
  class Impl;
 public:
  Runner();
  ~Runner();
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;
 public:
  eExecutionPolicy executionPolicy() const;
  void setExecutionPolicy(eExecutionPolicy v);
 private:
  // TODO: a supprimer
  RunQueueImpl* _internalCreateOrGetRunQueueImpl(eExecutionPolicy exec_policy);
  RunQueueImpl* _internalCreateOrGetRunQueueImpl(const RunQueueBuildInfo& bi);
  void _internalFreeRunQueueImpl(RunQueueImpl*);
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une file temporaire associée à \a runner.
inline RunQueue
makeQueue(Runner& runner)
{
  return RunQueue(runner);
}

//! Créé une file temporaire associée à \a runner.
inline RunQueue
makeQueue(Runner* runner)
{
  ARCANE_CHECK_POINTER(runner);
  return RunQueue(*runner);
}

//! Créé une file temporaire associée à \a runner avec les propriétés \a bi.
inline RunQueue
makeQueue(Runner& runner,const RunQueueBuildInfo& bi)
{
  return RunQueue(runner,bi);
}

//! Créé une file temporaire associée à \a runner avec les propriétés \a bi.
inline RunQueue
makeQueue(Runner* runner,const RunQueueBuildInfo& bi)
{
  ARCANE_CHECK_POINTER(runner);
  return RunQueue(*runner,bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une file avec la politique d'exécution \a exec_policy
ARCCORE_DEPRECATED_2021("Use a specific runner to change policy")
inline RunQueue
makeQueue(Runner& runner,eExecutionPolicy exec_policy)
{
  return RunQueue(runner,exec_policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Créé une référence sur file avec la politique d'exécution par défaut de \a runner.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
inline Ref<RunQueue>
makeQueueRef(Runner& runner)
{
  return makeRef(new RunQueue(runner));
}

/*!
 * \brief Créé une référence sur file avec la politique d'exécution par défaut de \a runner.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
inline Ref<RunQueue>
makeQueueRef(Runner& runner,const RunQueueBuildInfo& bi)
{
  return makeRef(new RunQueue(runner,bi));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Créé une référence sur file avec la politique d'exécution par défaut de \a runner.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
inline Ref<RunQueue>
makeQueueRef(Runner* runner)
{
  ARCANE_CHECK_POINTER(runner);
  return makeRef(new RunQueue(*runner));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Créé une référence sur une file avec la politique d'exécution \a exec_policy.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
ARCCORE_DEPRECATED_2021("Use a specific runner to change policy")
inline Ref<RunQueue>
makeQueueRef(Runner& runner,eExecutionPolicy exec_policy)
{
  return makeRef(new RunQueue(runner,exec_policy));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
