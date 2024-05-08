// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.cc                                                 (C) 2000-2024 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/RunQueueEvent.h"
#include "arcane/accelerator/core/internal/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"
#include "arcane/accelerator/core/internal/RunnerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(Runner& runner)
: m_p(impl::RunQueueImpl::create(runner._impl()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(Runner& runner, const RunQueueBuildInfo& bi)
: m_p(impl::RunQueueImpl::create(runner._impl(), bi))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(const RunQueue& x)
: m_p(x.m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(RunQueue&& x) noexcept
: m_p(x.m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue& RunQueue::
operator=(const RunQueue& x)
{
  if (&x != this)
    m_p = x.m_p;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue& RunQueue::
operator=(RunQueue&& x) noexcept
{
  m_p = x.m_p;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
~RunQueue()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
_checkNotNull() const
{
  if (!m_p)
    ARCANE_FATAL("Invalid operation on null RunQueue");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
barrier() const
{
  if (m_p)
    m_p->_internalBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eExecutionPolicy RunQueue::
executionPolicy() const
{
  if (m_p)
    return m_p->executionPolicy();
  return eExecutionPolicy::None;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunnerRuntime* RunQueue::
_internalRuntime() const
{
  _checkNotNull();
  return m_p->_internalRuntime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunQueueStream* RunQueue::
_internalStream() const
{
  _checkNotNull();
  return m_p->_internalStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunCommandImpl* RunQueue::
_getCommandImpl() const
{
  _checkNotNull();
  return m_p->_internalCreateOrGetRunCommandImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* RunQueue::
platformStream()
{
  if (m_p)
    return m_p->_internalStream()->_internalImpl();
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
copyMemory(const MemoryCopyArgs& args) const
{
  _checkNotNull();
  _internalStream()->copyMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
prefetchMemory(const MemoryPrefetchArgs& args) const
{
  _checkNotNull();
  _internalStream()->prefetchMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
waitEvent(RunQueueEvent& event)
{
  _checkNotNull();
  auto* p = event._internalEventImpl();
  return p->waitForEvent(_internalStream());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
waitEvent(Ref<RunQueueEvent>& event)
{
  _checkNotNull();
  waitEvent(*event.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
recordEvent(RunQueueEvent& event)
{
  _checkNotNull();
  auto* p = event._internalEventImpl();
  return p->recordQueue(_internalStream());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
recordEvent(Ref<RunQueueEvent>& event)
{
  _checkNotNull();
  recordEvent(*event.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
setAsync(bool v)
{
  _checkNotNull();
  m_p->m_is_async = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueue::
isAsync() const
{
  if (m_p)
    return m_p->m_is_async;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueue::
_isAutoPrefetchCommand() const
{
  _checkNotNull();
  return m_p->m_runner_impl->isAutoPrefetchCommand();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueue::
isAcceleratorPolicy() const
{
  return Arcane::Accelerator::isAcceleratorPolicy(executionPolicy());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions RunQueue::
allocationOptions() const
{
  if (m_p)
    return m_p->allocationOptions();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
setMemoryRessource(eMemoryRessource mem)
{
  _checkNotNull();
  m_p->m_memory_ressource = mem;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eMemoryRessource RunQueue::
memoryRessource() const
{
  if (m_p)
    return m_p->m_memory_ressource;
  return eMemoryRessource::Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr, PointerAttribute* ptr_attr)
{
  if (!queue || queue->isNull())
    return ePointerAccessibility::Unknown;
  return impl::RuntimeStaticInfo::getPointerAccessibility(queue->executionPolicy(), ptr, ptr_attr);
}

extern "C++" void impl::
arcaneCheckPointerIsAccessible(RunQueue* queue, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  if (!queue || queue->isNull())
    return;
  return impl::RuntimeStaticInfo::checkPointerIsAcccessible(queue->executionPolicy(), ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
