﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandImpl.h                                            (C) 2000-2023 */
/*                                                                           */
/* Implémentation de la gestion d'une commande sur accélérateur.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_RUNCOMMANDIMPL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_RUNCOMMANDIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ParallelLoopOptions.h"
#include "arcane/utils/Profiling.h"

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include <set>
#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'une commande pour accélérateur.
 */
class RunCommandImpl
{
  friend RunCommand;

 public:

  RunCommandImpl(RunQueueImpl* queue);
  ~RunCommandImpl();
  RunCommandImpl(const RunCommandImpl&) = delete;
  RunCommandImpl& operator=(const RunCommandImpl&) = delete;

 public:

  static RunCommandImpl* create(RunQueueImpl* r);

 public:

  const TraceInfo& traceInfo() const { return m_trace_info; }
  const String& kernelName() const { return m_kernel_name; }

 public:

  void notifyBeginLaunchKernel();
  void notifyEndLaunchKernel();
  void notifyEndExecuteKernel();
  impl::IReduceMemoryImpl* getOrCreateReduceMemoryImpl();

  void releaseReduceMemoryImpl(ReduceMemoryImpl* p);
  IRunQueueStream* internalStream() const;
  Runner* runner() const;

 private:

  ReduceMemoryImpl* _getOrCreateReduceMemoryImpl();

 private:

  RunQueueImpl* m_queue;
  TraceInfo m_trace_info;
  String m_kernel_name;
  Int32 m_nb_thread_per_block = 0;
  ParallelLoopOptions m_parallel_loop_options;

  // NOTE: cette pile gère la mémoire associé à un seul runtime
  // Si on souhaite un jour supporté plusieurs runtimes il faudra une pile
  // par runtime. On peut éventuellement limiter cela si on est sur
  // qu'une commande est associée à un seul type (au sens runtime) de RunQueue.
  std::stack<ReduceMemoryImpl*> m_reduce_memory_pool;

  //! Liste des réductions actives
  std::set<ReduceMemoryImpl*> m_active_reduce_memory_list;

  //! Indique si la commande a été lancée.
  bool m_has_been_launched = false;

  //! Indique si on utilise les évènements séquentiels pour calculer le temps d'exécution
  bool m_use_sequential_timer_event = false;
  //! Evènements pour le début et la fin de l'exécution.
  IRunQueueEventImpl* m_start_event = nullptr;
  //! Evènements pour la fin de l'exécution.
  IRunQueueEventImpl* m_stop_event = nullptr;

  //! Temps au lancement de la commande
  Int64 m_begin_time = 0;

  ForLoopOneExecStat m_loop_one_exec_stat;
  ForLoopOneExecStat* m_loop_one_exec_stat_ptr = nullptr;

  //! Indique si la commande s'exécute sur accélérateur
  const bool m_use_accelerator = false;

 private:

  void _freePools();
  void _reset();
  void _init();
  IRunQueueEventImpl* _createEvent();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
