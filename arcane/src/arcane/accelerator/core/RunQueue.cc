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
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/RunQueueEvent.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(Runner& runner)
: m_p(impl::RunQueueImpl::create(&runner))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(Runner& runner, const RunQueueBuildInfo& bi)
: m_p(impl::RunQueueImpl::create(&runner, bi))
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

RunQueue& RunQueue::
operator=(const RunQueue& x)
{
  if (&x != this)
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
barrier()
{
  m_p->_internalBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eExecutionPolicy RunQueue::
executionPolicy() const
{
  return m_p->executionPolicy();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunnerRuntime* RunQueue::
_internalRuntime() const
{
  return m_p->_internalRuntime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IRunQueueStream* RunQueue::
_internalStream() const
{
  return m_p->_internalStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunCommandImpl* RunQueue::
_getCommandImpl()
{
  return m_p->_internalCreateOrGetRunCommandImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* RunQueue::
platformStream()
{
  return m_p->_internalStream()->_internalImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
copyMemory(const MemoryCopyArgs& args)
{
  _internalStream()->copyMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
prefetchMemory(const MemoryPrefetchArgs& args)
{
  _internalStream()->prefetchMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
waitEvent(RunQueueEvent& event)
{
  auto* p = event._internalEventImpl();
  return p->waitForEvent(_internalStream());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
waitEvent(Ref<RunQueueEvent>& event)
{
  waitEvent(*event.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
recordEvent(RunQueueEvent& event)
{
  auto* p = event._internalEventImpl();
  return p->recordQueue(_internalStream());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
recordEvent(Ref<RunQueueEvent>& event)
{
  recordEvent(*event.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
setAsync(bool v)
{
  m_p->m_is_async = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueue::
isAsync() const
{
  return m_p->m_is_async;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueue::
_isAutoPrefetchCommand() const
{
  return m_p->m_runner->_isAutoPrefetchCommand();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr, PointerAttribute* ptr_attr)
{
  if (!queue)
    return ePointerAccessibility::Unknown;
  return impl::RuntimeStaticInfo::getPointerAccessibility(queue->executionPolicy(), ptr, ptr_attr);
}

extern "C++" void impl::
arcaneCheckPointerIsAcccessible(RunQueue* queue, const void* ptr,
                                const char* name, const TraceInfo& ti)
{
  if (!queue)
    return;
  return impl::RuntimeStaticInfo::checkPointerIsAcccessible(queue->executionPolicy(), ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
