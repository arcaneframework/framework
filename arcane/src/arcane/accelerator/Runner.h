// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.h                                                    (C) 2000-2021 */
/*                                                                           */
/* Gestion de l'exécution sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNNER_H
#define ARCANE_ACCELERATOR_RUNNER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class AcceleratorRuntimeInitialisationInfo;
}

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
class ARCANE_ACCELERATOR_EXPORT Runner
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
  RunQueueImpl* _internalCreateOrGetRunQueueImpl(eExecutionPolicy exec_policy);
  void _internalFreeRunQueueImpl(RunQueueImpl*);
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une file avec la politique d'exécution par défaut de \a runner.
inline RunQueue
makeQueue(Runner& runner)
{
  return RunQueue(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une file avec la politique d'exécution \a exec_policy
inline RunQueue
makeQueue(Runner& runner,eExecutionPolicy exec_policy)
{
  return RunQueue(runner,exec_policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Initialise \a runner en fonction de
 * la valeur de \a acc_info.
 */
extern "C++" ARCANE_ACCELERATOR_EXPORT void
initializeRunner(Runner& runner,ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
