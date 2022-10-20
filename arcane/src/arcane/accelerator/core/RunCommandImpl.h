// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandImpl.h                                            (C) 2000-2022 */
/*                                                                           */
/* Implémentation de la gestion d'une commande sur accélérateur.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNCOMMANDIMPL_H
#define ARCANE_ACCELERATOR_CORE_RUNCOMMANDIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ConcurrencyUtils.h"

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

  void reset();
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

 private:

  void _freePools();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
