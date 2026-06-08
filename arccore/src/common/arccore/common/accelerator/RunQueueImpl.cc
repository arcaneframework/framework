// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Management of a run queue on an accelerator.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/internal/RunQueueImpl.h"

#include "arccore/common/MemoryUtils.h"
#include "arccore/common/SmallArray.h"

#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/IRunQueueStream.h"
#include "arccore/common/accelerator/internal/RunCommandImpl.h"
#include "arccore/common/accelerator/internal/RunnerImpl.h"
#include "arccore/common/accelerator/internal/IRunQueueEventImpl.h"

#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/DeviceId.h"
#include "arccore/common/accelerator/RunQueueEvent.h"
#include "arccore/common/accelerator/KernelLaunchArgs.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Lock for the RunCommand pool in multi-thread.
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
  bool is_check = arccoreIsCheck();
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
  // If there are commands currently running when releasing
  // the run queue, we must wait to avoid memory leaks because
  // the commands will not be deallocated.
  // TODO: Check if it should rather indicate this to the user
  // or throw a fatal error.
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
  m_memory_ressource = eMemoryResource::Host;
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
 * \brief Frees running commands.
 *
 * This method is generally called after a barrier, which guarantees
 * that asynchronous commands are finished.
 */
void RunQueueImpl::
_internalFreeRunningCommands()
{
  if (m_use_pool_mutex) {
    SmallArray<RunCommandImpl*> command_list;
    // Copy the commands into a local array because m_active_run_command_list
    // may be modified by another thread.
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
 * \brief Returns the command to the pool if possible.
 *
 * The command is not returned to the pool as long as there is a RunCommand
 * referencing it. In this case, the command will be returned to the pool
 * when the RunCommand destructor is called. This is necessary to
 * handle the case where a RunCommand is created but never used because
 * in this case it will never be in m_active_run_command_list and will not
 * be processed when calling _internalFreeRunningCommands().
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
 * \brief Blocks until all commands are finished.
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
 * \brief Resets the implementation
 *
 * This method is called when initializing a RunQueue with
 * this instance. In this case, the instance values
 * that depend on the current state must be reset.
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

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
