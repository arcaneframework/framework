// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandImpl.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Implémentation de la gestion d'une commande sur accélérateur.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/internal/RunCommandImpl.h"
#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ForLoopTraceInfo.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/Convert.h"
#include "arccore/base/ConcurrencyBase.h"

#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/internal/IRunQueueEventImpl.h"
#include "arccore/common/accelerator/internal/IRunQueueStream.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"
#include "arccore/common/accelerator/internal/ReduceMemoryImpl.h"
#include "arccore/common/accelerator/internal/RunnerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

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

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_ALLOW_REUSE_COMMAND", true))
    m_is_allow_reuse_command = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl* RunCommandImpl::
create(RunQueueImpl* r)
{
  RunCommandImpl* c = r->_internalCreateOrGetRunCommandImpl();
  c->_reset();
  return c;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notification du début d'exécution de la commande.
 */
void RunCommandImpl::
notifyBeginLaunchKernel()
{
  if (m_has_been_launched) {
    if (!m_is_allow_reuse_command)
      ARCCORE_FATAL("Command has already been launched. You can not re-use the same command.\n"
                    "  You can temporarily allow it if you set environment variable\n"
                    "    ARCANE_ACCELERATOR_ALLOW_REUSE_COMMAND to 1\n");
  }
  IRunQueueStream* stream = internalStream();
  stream->notifyBeginLaunchKernel(*this);
  // TODO: utiliser la bonne stream en séquentiel
  m_has_been_launched = true;
  if (m_use_profiling) {
    m_start_event->recordQueue(stream);
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
  if (m_use_profiling)
    m_stop_event->recordQueue(stream);
  stream->notifyEndLaunchKernel(*this);
  m_queue->_addRunningCommand(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notification du lancement d'un kernel SYCL.
 *
 * \a sycl_event_ptr est de type sycl::event* et contient
 * l'évènement associé à la commande qui est retourné lors
 * des appels à sycl::queue::submit().
 */
void RunCommandImpl::
notifyLaunchKernelSyclEvent(void* sycl_event_ptr)
{
  IRunQueueStream* stream = internalStream();
  stream->_setSyclLastCommandEvent(sycl_event_ptr);
  // Il faut enregistrer à nouveau la file associée à l'évènement
  // car lors de l'appel à notifyBeginLaunchKernel() il n'y avait pas
  // encore l'évènement associé à cette file.
  m_start_event->recordQueue(stream);
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

  Int64 diff_time_ns = 0;
  if (m_use_profiling){
    diff_time_ns = m_stop_event->elapsedTime(m_start_event);
    runner()->addTime((double)diff_time_ns / 1.0e9);
  }

  ForLoopOneExecStat* exec_info = m_loop_one_exec_stat_ptr;
  if (exec_info) {
    exec_info->setEndTime(m_begin_time + diff_time_ns);
    //std::cout << "END_EXEC exec_info=" << m_loop_run_info.traceInfo().traceInfo() << "\n";
    ForLoopTraceInfo flti(traceInfo(), kernelName());
    ProfilingRegistry::_threadLocalForLoopInstance()->merge(*exec_info, flti);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_reset()
{
  m_kernel_name = String();
  m_trace_info = TraceInfo();
  m_nb_thread_per_block = 0;
  m_use_profiling = ProfilingRegistry::hasProfiling();
  m_parallel_loop_options = ConcurrencyBase::defaultParallelLoopOptions();
  m_begin_time = 0;
  m_loop_one_exec_stat.reset();
  m_loop_one_exec_stat_ptr = nullptr;
  m_has_been_launched = false;
  m_has_living_run_command = false;
  m_may_be_put_in_pool = false;
  m_shared_memory_size = 0;
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
    ARCCORE_FATAL("ReduceMemoryImpl in not in active list");
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
/*!
 * \brief Méthode appelée quand l'instance RunCommand associée est détruite.
 */
void RunCommandImpl::
_notifyDestroyRunCommand()
{
  // Si la commande n'a pas été lancé, il faut la remettre dans le pool
  // des commandes de la file (sinon on aura une fuite mémoire)
  if (!m_has_been_launched || m_may_be_put_in_pool)
    m_queue->_putInCommandPool(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
