// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"
#include "arccore/common/accelerator/RunQueue.h"

#include "arccore/base/FatalErrorException.h"

#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/IRunQueueStream.h"
#include "arccore/common/accelerator/internal/IRunQueueEventImpl.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"
#include "arccore/common/accelerator/internal/RunnerImpl.h"
#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/RunQueueEvent.h"
#include "arccore/common/accelerator/Memory.h"
#include "arccore/common/accelerator/NativeStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE : Les constructeurs et destructeurs doivent être dans le fichier source,
// car le type \a m_p est opaque pour l'utilisation n'est pas connu dans
// la définition de la classe.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(const Runner& runner)
: m_p(Impl::RunQueueImpl::create(runner._impl()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(const Runner& runner, const RunQueueBuildInfo& bi)
: m_p(Impl::RunQueueImpl::create(runner._impl(), bi))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(const Runner& runner, bool)
: m_p(Impl::RunQueueImpl::create(runner._impl()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(const Runner& runner, const RunQueueBuildInfo& bi, bool)
: m_p(Impl::RunQueueImpl::create(runner._impl(), bi))
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
: m_p(std::move(x.m_p))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue::
RunQueue(Impl::RunQueueImpl* p)
: m_p(p)
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
  m_p = std::move(x.m_p);
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
    ARCCORE_FATAL("Invalid operation on null RunQueue");
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

Impl::IRunnerRuntime* RunQueue::
_internalRuntime() const
{
  return m_p->_internalRuntime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::IRunQueueStream* RunQueue::
_internalStream() const
{
  return m_p->_internalStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::RunCommandImpl* RunQueue::
_getCommandImpl() const
{
  return m_p->_internalCreateOrGetRunCommandImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::RunQueueImpl* RunQueue::
_internalImpl() const
{
  _checkNotNull();
  return m_p.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::NativeStream RunQueue::
_internalNativeStream() const
{
  if (m_p)
    return m_p->_internalStream()->nativeStream();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* RunQueue::
platformStream() const
{
  return _internalNativeStream().m_native_pointer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
copyMemory(const MemoryCopyArgs& args) const
{
  _checkNotNull();
  m_p->copyMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
prefetchMemory(const MemoryPrefetchArgs& args) const
{
  _checkNotNull();
  m_p->prefetchMemory(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
waitEvent(RunQueueEvent& event)
{
  _checkNotNull();
  m_p->waitEvent(event);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
waitEvent(Ref<RunQueueEvent>& event)
{
  RunQueueEvent* e = event.get();
  ARCCORE_CHECK_POINTER(e);
  waitEvent(*e);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
recordEvent(RunQueueEvent& event)
{
  _checkNotNull();
  m_p->recordEvent(event);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
recordEvent(Ref<RunQueueEvent>& event)
{
  RunQueueEvent* e = event.get();
  ARCCORE_CHECK_POINTER(e);
  recordEvent(*e);
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

const RunQueue& RunQueue::
addAsync(bool is_async) const
{
  _checkNotNull();
  m_p->m_is_async = is_async;
  return (*this);
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
setMemoryRessource(eMemoryResource mem)
{
  _checkNotNull();
  m_p->m_memory_ressource = mem;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eMemoryResource RunQueue::
memoryRessource() const
{
  if (m_p)
    return m_p->m_memory_ressource;
  return eMemoryResource::Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eMemoryResource RunQueue::
memoryResource() const
{
  if (m_p)
    return m_p->m_memory_ressource;
  return eMemoryResource::Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueue::
setConcurrentCommandCreation(bool v)
{
  _checkNotNull();
  if (isAcceleratorPolicy())
    ARCCORE_FATAL("setting concurrent command creation is not supported for RunQueue running on accelerator");
  m_p->setConcurrentCommandCreation(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueue::
isConcurrentCommandCreation() const
{
  if (m_p)
    return m_p->isConcurrentCommandCreation();
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr, PointerAttribute* ptr_attr)
{
  if (!queue || queue->isNull())
    return ePointerAccessibility::Unknown;
  return Impl::RuntimeStaticInfo::getPointerAccessibility(queue->executionPolicy(), ptr, ptr_attr);
}

extern "C++" void Impl::
arcaneCheckPointerIsAccessible(const RunQueue* queue, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  if (!queue || queue->isNull())
    return;
  return Impl::RuntimeStaticInfo::checkPointerIsAcccessible(queue->executionPolicy(), ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
