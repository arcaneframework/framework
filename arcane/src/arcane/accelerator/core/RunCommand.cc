// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.cc                                               (C) 2000-2021 */
/*                                                                           */
/* Gestion d'une commande sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunCommand.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IMemoryAllocator.h"

#include "arcane/accelerator/core/RunQueueImpl.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/IReduceMemoryImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file RunCommandLoop.h
 *
 * \brief Types et macros pour gérer les boucles sur les accélérateurs
 */

/*!
 * \file RunCommandEnumerate.h
 *
 * \brief Types et macros pour gérer les énumérations des entités sur les accélérateurs
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ReduceMemoryImpl
: public IReduceMemoryImpl
{
 public:
  ReduceMemoryImpl(RunCommandImpl* p)
  : m_command(p)
  {
    // TODO: prendre en compte la politique d'exécution pour savoir
    // comment allouer.
    m_allocator = platform::getAcceleratorHostMemoryAllocator();
    if (!m_allocator)
      ARCANE_FATAL("No HostMemory allocator available for accelerator");
    m_size = 128;
    m_managed_memory = m_allocator->allocate(m_size);
  }
  ~ReduceMemoryImpl()
  {
    m_allocator->deallocate(m_managed_memory);
  }
 public:
  void* allocateMemory(Int64 s) override
  {
    if (s>m_size){
      m_managed_memory = m_allocator->reallocate(m_managed_memory,s);
      m_size = s;
    }
    return m_managed_memory;
  }
  void release() override;
 private:
  RunCommandImpl* m_command;
  IMemoryAllocator* m_allocator;
  void* m_managed_memory = nullptr; //! Pointeur sur la mémoire allouée
  Int64 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'une commande pour accélérateur.
 * \warning API en cours de définition.
 */
class RunCommandImpl
{
  friend class RunCommand;
 public:
  RunCommandImpl(RunQueueImpl* queue);
  ~RunCommandImpl();
  RunCommandImpl(const RunCommandImpl&) = delete;
  RunCommandImpl& operator=(const RunCommandImpl&) = delete;
 public:
  static RunCommandImpl* create(RunQueueImpl* r);
 public:
  void release();
  const TraceInfo& traceInfo() const { return m_trace_info; }
  const String& kernelName() const { return m_kernel_name; }
 public:
  void reset();
  impl::IReduceMemoryImpl* getOrCreateReduceMemoryImpl()
  {
    // Pas besoin d'allouer de la mémoire spécifique si on n'est pas
    // sur un accélérateur
    if (!impl::isAcceleratorPolicy(m_queue->executionPolicy()))
      return nullptr;

    auto& pool = m_reduce_memory_pool;

    if (!pool.empty()){
      ReduceMemoryImpl* p = pool.top();
      pool.pop();
      return p;
    }
    return new ReduceMemoryImpl(this);
  }
  void releaseReduceMemoryImpl(ReduceMemoryImpl* p)
  {
    m_reduce_memory_pool.push(p);
  }
 public:
  RunQueueImpl* m_queue;
  TraceInfo m_trace_info;
  String m_kernel_name;
  // NOTE: cette pile gère la mémoire associé à un seul runtime
  // Si on souhaite un jour supporté plusieurs runtimes il faudra une pile
  // par runtime. On peut éventuellement limiter cela si on est sur
  // qu'une commande est associée à un seul type (au sens runtime) de RunQueue.
  std::stack<ReduceMemoryImpl*> m_reduce_memory_pool;
 private:
  void _freePools();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
RunCommandImpl(RunQueueImpl* queue)
: m_queue(queue)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
~RunCommandImpl()
{
  _freePools();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_freePools()
{
  while (!m_reduce_memory_pool.empty()){
    delete m_reduce_memory_pool.top();
    m_reduce_memory_pool.pop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
release()
{
  m_queue->_internalFreeRunCommandImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl* RunCommandImpl::
create(RunQueueImpl* r)
{
  return r->_internalCreateOrGetRunCommandImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
reset()
{
  m_kernel_name = String();
  m_trace_info = TraceInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
release()
{
  m_command->releaseReduceMemoryImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand::
RunCommand(RunQueue& run_queue)
: m_run_queue(run_queue)
, m_p(run_queue._getCommandImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand::
~RunCommand()
{
  m_p->release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_resetInfos()
{
  m_p->reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const TraceInfo& RunCommand::
traceInfo() const
{
  return m_p->traceInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& RunCommand::
kernelName() const
{
  return m_p->kernelName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addTraceInfo(const TraceInfo& ti)
{
  m_p->m_trace_info = ti;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addKernelName(const String& v)
{
  m_p->m_kernel_name = v;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
RunCommand& operator<<(RunCommand& command,const TraceInfo& trace_info)
{
  return command.addTraceInfo(trace_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunCommandImpl* RunCommand::
_internalCreateImpl(impl::RunQueueImpl* queue)
{
  return new impl::RunCommandImpl(queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalDestroyImpl(impl::RunCommandImpl* p)
{
  delete p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
extern "C++" IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command)
{
  return command->m_p->getOrCreateReduceMemoryImpl();
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
