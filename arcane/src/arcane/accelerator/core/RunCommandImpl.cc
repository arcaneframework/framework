// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandImpl.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Implémentation de la gestion d'une commande sur accélérateur.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"

#include "arcane/utils/ForLoopTraceInfo.h"
#include "arcane/utils/ConcurrencyUtils.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"
#include "arcane/accelerator/core/internal/ReduceMemoryImpl.h"
#include "arcane/accelerator/core/internal/RunnerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
RunCommandImpl(RunQueueImpl* queue)
: m_queue(queue)
, m_use_accelerator(impl::isAcceleratorPolicy(queue->runner()->executionPolicy()))
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
~RunCommandImpl()
{
  _freePools();
  delete m_start_event;
  delete m_stop_event;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_freePools()
{
  while (!m_reduce_memory_pool.empty()) {
    delete m_reduce_memory_pool.top();
    m_reduce_memory_pool.pop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IRunQueueEventImpl* RunCommandImpl::
_createEvent()
{
  if (m_use_sequential_timer_event)
    return getSequentialRunQueueRuntime()->createEventImplWithTimer();
  return runner()->_createEventWithTimer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_init()
{
  // N'utilise les timers accélérateur que si le profiling est activé.
  // On fait cela pour éviter d'appeler les évènements accélérateurs car on
  // ne connait pas encore leur influence sur les performances. Si elle est
  // négligeable alors on pourra l'activer par défaut.

  // TODO: il faudrait éventuellement avoir une instance séquentielle et
  // une associée à runner() pour gérer le cas ou ProfilingRegistry::hasProfiling()
  // change en cours d'exécution.
  if (m_use_accelerator && !ProfilingRegistry::hasProfiling())
    m_use_sequential_timer_event = true;

  m_start_event = _createEvent();
  m_stop_event = _createEvent();
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
/*!
 * \brief Notification du début d'exécution de la commande.
 */
void RunCommandImpl::
notifyBeginLaunchKernel()
{
  IRunQueueStream* stream = internalStream();
  stream->notifyBeginLaunchKernel(*this);
  // TODO: utiliser la bonne stream en séquentiel
  m_start_event->recordQueue(stream);
  m_has_been_launched = true;
  if (ProfilingRegistry::hasProfiling()) {
    m_begin_time = platform::getRealTimeNS();
    m_loop_one_exec_stat_ptr = &m_loop_one_exec_stat;
    m_loop_one_exec_stat.setBeginTime(m_begin_time);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notification de la fin de lancement de la commande.
 *
 * La commande continue à s'exécuter en tâche de fond.
 */
void RunCommandImpl::
notifyEndLaunchKernel()
{
  IRunQueueStream* stream = internalStream();
  // TODO: utiliser la bonne stream en séquentiel
  m_stop_event->recordQueue(stream);
  stream->notifyEndLaunchKernel(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notification de la fin d'exécution du noyau.
 *
 * Après cet appel, on est sur que la commande a fini de s'exécuter et on
 * peut la recycler. En asynchrone, cette méthode est appelée lors de la
 * synchronisation d'une file.
 */
void RunCommandImpl::
notifyEndExecuteKernel()
{
  // Ne fait rien si la commande n'a pas été lancée.
  if (!m_has_been_launched)
    return;

  Int64 diff_time_ns = m_stop_event->elapsedTime(m_start_event);

  runner()->addTime((double)diff_time_ns / 1.0e9);

  ForLoopOneExecStat* exec_info = m_loop_one_exec_stat_ptr;
  if (exec_info) {
    exec_info->setEndTime(m_begin_time + diff_time_ns);
    //std::cout << "END_EXEC exec_info=" << m_loop_run_info.traceInfo().traceInfo() << "\n";
    ForLoopTraceInfo flti(traceInfo(), kernelName());
    ProfilingRegistry::_threadLocalForLoopInstance()->merge(*exec_info, flti);
  }

  _reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_reset()
{
  m_kernel_name = String();
  m_trace_info = TraceInfo();
  m_nb_thread_per_block = 0;
  m_parallel_loop_options = TaskFactory::defaultParallelLoopOptions();
  m_begin_time = 0;
  m_loop_one_exec_stat.reset();
  m_loop_one_exec_stat_ptr = nullptr;
  m_has_been_launched = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IReduceMemoryImpl* RunCommandImpl::
getOrCreateReduceMemoryImpl()
{
  ReduceMemoryImpl* p = _getOrCreateReduceMemoryImpl();
  if (p) {
    m_active_reduce_memory_list.insert(p);
  }
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
releaseReduceMemoryImpl(ReduceMemoryImpl* p)
{
  auto x = m_active_reduce_memory_list.find(p);
  if (x == m_active_reduce_memory_list.end())
    ARCANE_FATAL("ReduceMemoryImpl in not in active list");
  m_active_reduce_memory_list.erase(x);
  m_reduce_memory_pool.push(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IRunQueueStream* RunCommandImpl::
internalStream() const
{
  return m_queue->_internalStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunnerImpl* RunCommandImpl::
runner() const
{
  return m_queue->runner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReduceMemoryImpl* RunCommandImpl::
_getOrCreateReduceMemoryImpl()
{
  // Pas besoin d'allouer de la mémoire spécifique si on n'est pas
  // sur un accélérateur
  if (!m_use_accelerator)
    return nullptr;

  auto& pool = m_reduce_memory_pool;

  if (!pool.empty()) {
    ReduceMemoryImpl* p = pool.top();
    pool.pop();
    return p;
  }
  return new ReduceMemoryImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
