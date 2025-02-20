// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/RunQueueImpl.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/SmallArray.h"

#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/RunnerImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/accelerator/core/RunQueueEvent.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Verrou pour le pool de RunCommand en multi-thread.
class RunQueueImpl::Lock
{
 public:

  explicit Lock(RunQueueImpl* p)
  {
    if (p->m_use_pool_mutex) {
      m_mutex = p->m_pool_mutex.get();
      if (m_mutex) {
        m_mutex->lock();
      }
    }
  }
  ~Lock()
  {
    if (m_mutex)
      m_mutex->unlock();
  }
  Lock(const Lock&) = delete;
  Lock& operator=(const Lock&) = delete;

 private:

  std::mutex* m_mutex = nullptr;
};

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
  delete m_queue_stream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_freeCommandsInPool()
{
  bool is_check = arcaneIsCheck();
  std::unordered_set<RunCommandImpl*> command_set;
  while (!m_run_command_pool.empty()) {
    RunCommandImpl* c = m_run_command_pool.top();
    if (is_check) {
      if (command_set.find(c) != command_set.end())
        std::cerr << "Command is present several times in the command pool\n";
      command_set.insert(c);
    }
    RunCommand::_internalDestroyImpl(c);
    m_run_command_pool.pop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_destroy(RunQueueImpl* q)
{
  q->_freeCommandsInPool();
  delete q;
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
    RunQueueImpl::_destroy(this);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_setDefaultMemoryRessource()
{
  m_memory_ressource = eMemoryRessource::Host;
  if (isAcceleratorPolicy(m_execution_policy))
    m_memory_ressource = MemoryUtils::getDefaultDataMemoryResource();
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
  RunCommandImpl* p = nullptr;

  {
    auto& pool = m_run_command_pool;
    Lock my_lock(this);
    if (!pool.empty()) {
      p = pool.top();
      pool.pop();
    }
  }
  if (!p)
    p = RunCommand::_internalCreateImpl(this);
  p->_reset();
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
  if (m_use_pool_mutex) {
    SmallArray<RunCommandImpl*> command_list;
    // Recopie les commandes dans un tableau local car m_active_run_command_list
    // peut être modifié par un autre thread.
    {
      Lock my_lock(this);
      for (RunCommandImpl* p : m_active_run_command_list) {
        command_list.add(p);
      }
      m_active_run_command_list.clear();
    }
    for (RunCommandImpl* p : command_list) {
      p->notifyEndExecuteKernel();
    }
    {
      Lock my_lock(this);
      for (RunCommandImpl* p : command_list) {
        _checkPutCommandInPoolNoLock(p);
      }
    }
  }
  else {
    for (RunCommandImpl* p : m_active_run_command_list) {
      p->notifyEndExecuteKernel();
      _checkPutCommandInPoolNoLock(p);
    }
    m_active_run_command_list.clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remet la commande dans le pool si possible.
 *
 * On ne remet pas la commande dans le pool tant qu'il y a une RunCommand
 * qui y fait référence. Dans ce cas la commande sera remise dans le pool
 * lors de l'appel au destructeur de RunCommand. Cela est nécessaire pour
 * gérer le cas où une RunCommand est créée mais n'est jamais utilisée car
 * dans ce cas elle ne sera jamais dans m_active_run_command_list et ne
 * sera pas traitée lors de l'appel à _internalFreeRunningCommands().
 */
void RunQueueImpl::
_checkPutCommandInPoolNoLock(RunCommandImpl* p)
{
  if (p->m_has_living_run_command)
    p->m_may_be_put_in_pool = true;
  else
    m_run_command_pool.push(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_addRunningCommand(RunCommandImpl* p)
{
  Lock my_lock(this);
  m_active_run_command_list.add(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_putInCommandPool(RunCommandImpl* p)
{
  Lock my_lock(this);
  m_run_command_pool.push(p);
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

void RunQueueImpl::
setConcurrentCommandCreation(bool v)
{
  m_use_pool_mutex = v;
  if (!m_pool_mutex.get())
    m_pool_mutex = std::make_unique<std::mutex>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
dumpStats(std::ostream& ostr) const
{
  ostr << "nb_pool=" << m_run_command_pool.size()
       << " nb_active=" << m_active_run_command_list.size() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
