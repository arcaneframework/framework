// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.cc                                             (C) 2000-2023 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/RunQueueImpl.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl::
RunQueueImpl(Runner* runner, Int32 id, const RunQueueBuildInfo& bi)
: m_runner(runner)
, m_execution_policy(runner->executionPolicy())
, m_runtime(runner->_internalRuntime())
, m_queue_stream(m_runtime->createStream(bi))
, m_id(id)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl::
~RunQueueImpl()
{
  while (!m_run_command_pool.empty()) {
    RunCommand::_internalDestroyImpl(m_run_command_pool.top());
    m_run_command_pool.pop();
  }
  delete m_queue_stream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_release()
{
  // S'il reste des commandes en cours d'exécution au moment de libérer
  // la file d'exécution il faut attendre pour éviter des fuites mémoire car
  // les commandes ne seront pas désallouées.
  // TODO: Regarder s'il ne faudrait pas plutôt indiquer cela à l'utilisateur
  // ou faire une erreur fatale.
  if (!m_active_run_command_list.empty()){
    if (!_internalStream()->_barrierNoException()){
      _internalFreeRunningCommands();
    }
    else
      std::cerr << "WARNING: Error in internal accelerator barrier\n";
  }
  if (_isInPool())
    m_runner->_internalPutRunQueueImplInPool(this);
  else
    delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImpl::
create(Runner* r)
{
  return _reset(r->_internalCreateOrGetRunQueueImpl());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImpl::
create(Runner* r, const RunQueueBuildInfo& bi)
{
  return _reset(r->_internalCreateOrGetRunQueueImpl(bi));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl* RunQueueImpl::
_internalCreateOrGetRunCommandImpl()
{
  auto& pool = m_run_command_pool;
  RunCommandImpl* p = nullptr;

  // TODO: rendre thread-safe
  if (!pool.empty()) {
    p = pool.top();
    pool.pop();
  }
  else {
    p = RunCommand::_internalCreateImpl(this);
  }
  m_active_run_command_list.add(p);
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Libère les commandes en cours d'exécution.
 *
 * Cette méthode est en général appelée après une barrière ce qui garantit
 * que les commandes asynchrones sont terminées.
 */
void RunQueueImpl::
_internalFreeRunningCommands()
{
  for (RunCommandImpl* p : m_active_run_command_list) {
    p->notifyEndExecuteKernel();
    m_run_command_pool.push(p);
  }
  m_active_run_command_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Bloque jusqu'à ce que toutes les commandes soient terminées.
 */
void RunQueueImpl::
_internalBarrier()
{
  _internalStream()->barrier();
  _internalFreeRunningCommands();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Réinitialise l'implémentation
 *
 * Cette méthode est appelée lorsqu'on va initialiser une RunQueue avec
 * cette instance. Il faut dans ce car réinitialiser les valeurs de l'instance
 * qui dépendent de l'état actuel.
 */
RunQueueImpl* RunQueueImpl::
_reset(RunQueueImpl* p)
{
  p->m_nb_ref = 1;
  p->m_is_async = false;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
