// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/RunQueueImpl.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/RunnerImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/accelerator/core/RunQueueEvent.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl::
RunQueueImpl(RunnerImpl* runner_impl, Int32 id, const RunQueueBuildInfo& bi)
: m_runner_impl(runner_impl)
, m_execution_policy(runner_impl->executionPolicy())
, m_runtime(runner_impl->runtime())
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
  if (!m_active_run_command_list.empty()) {
    if (!_internalStream()->_barrierNoException()) {
      _internalFreeRunningCommands();
    }
    else
      std::cerr << "WARNING: Error in internal accelerator barrier\n";
  }
  if (_isInPool())
    m_runner_impl->_internalPutRunQueueImplInPool(this);
  else {
    delete this;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_setDefaultMemoryRessource()
{
  m_memory_ressource = eMemoryRessource::Host;
  if (isAcceleratorPolicy(m_execution_policy))
    m_memory_ressource = eMemoryRessource::UnifiedMemory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions RunQueueImpl::
allocationOptions() const
{
  MemoryAllocationOptions opt = MemoryUtils::getAllocationOptions(m_memory_ressource);
  Int16 device_id = static_cast<Int16>(m_runner_impl->deviceId().asInt32());
  opt.setDevice(device_id);
  return opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueueImpl::
isAutoPrefetchCommand() const
{
  return m_runner_impl->isAutoPrefetchCommand();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
copyMemory(const MemoryCopyArgs& args) const
{
  _internalStream()->copyMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
prefetchMemory(const MemoryPrefetchArgs& args) const
{
  _internalStream()->prefetchMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
recordEvent(RunQueueEvent& event)
{
  auto* p = event._internalEventImpl();
  return p->recordQueue(_internalStream());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
waitEvent(RunQueueEvent& event)
{
  auto* p = event._internalEventImpl();
  return p->waitForEvent(_internalStream());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImpl::
create(RunnerImpl* r)
{
  return _reset(r->_internalCreateOrGetRunQueueImpl());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImpl::
create(RunnerImpl* r, const RunQueueBuildInfo& bi)
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
  p->_reset();
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
 * cette instance. Il faut dans ce cas réinitialiser les valeurs de l'instance
 * qui dépendent de l'état actuel.
 */
RunQueueImpl* RunQueueImpl::
_reset(RunQueueImpl* p)
{
  p->m_is_async = false;
  p->_setDefaultMemoryRessource();
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
