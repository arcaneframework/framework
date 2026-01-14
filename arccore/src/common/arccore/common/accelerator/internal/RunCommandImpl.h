// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandImpl.h                                            (C) 2000-2026 */
/*                                                                           */
/* Implémentation de la gestion d'une commande sur accélérateur.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_RUNCOMMANDIMPL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_RUNCOMMANDIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/TraceInfo.h"
#include "arccore/base/Profiling.h"
#include "arccore/base/String.h"
#include "arccore/base/ParallelLoopOptions.h"

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include <set>
#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
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
  friend RunQueueImpl;

 public:

  explicit RunCommandImpl(RunQueueImpl* queue);
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
  Impl::IReduceMemoryImpl* getOrCreateReduceMemoryImpl();
  void releaseReduceMemoryImpl(ReduceMemoryImpl* p);
  IRunQueueStream* internalStream() const;
  RunnerImpl* runner() const;

 public:

  void notifyLaunchKernelSyclEvent(void* sycl_event_ptr);

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

  //! Indique si on souhaite le profiling
  bool m_use_profiling = false;

  //! Indique si on utilise les évènements séquentiels pour calculer le temps d'exécution
  bool m_use_sequential_timer_event = false;

  //! Évènements pour le début et la fin de l'exécution.
  IRunQueueEventImpl* m_start_event = nullptr;
  //! Évènements pour la fin de l'exécution.
  IRunQueueEventImpl* m_stop_event = nullptr;

  //! Temps au lancement de la commande
  Int64 m_begin_time = 0;

  ForLoopOneExecStat m_loop_one_exec_stat;
  ForLoopOneExecStat* m_loop_one_exec_stat_ptr = nullptr;

  //! Indique si la commande s'exécute sur accélérateur
  const bool m_use_accelerator = false;

  /*!
   * \brief Indique si on autorise à utiliser plusieurs fois la même commande.
   *
   * Normalement cela est interdit mais avant novembre 2024, il n'y avait pas
   * de mécanisme pour détecter cela. On peut donc temporairement autoriser
   * cela et dans un on supprimera cette possibilité.
   */
  bool m_is_allow_reuse_command = false;

  //! Indique si une RunCommand a une référence sur cette instance.
  bool m_has_living_run_command = false;

  //! Indique si on peut remettre la commande dans le pool associé à la RunQueue.
  bool m_may_be_put_in_pool = false;

  //! Taille de la mémoire partagée à allouer
  Int32 m_shared_memory_size = 0;

  //! Nombre de pas de décomposition de la boucle
  Int32 m_nb_stride = 1;

 private:

  void _freePools();
  void _reset();
  void _init();
  IRunQueueEventImpl* _createEvent();
  void _notifyDestroyRunCommand();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
