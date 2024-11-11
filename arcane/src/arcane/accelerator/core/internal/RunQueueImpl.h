// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.h                                              (C) 2000-2024 */
/*                                                                           */
/* Implémentation d'une 'RunQueue'.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_RUNQUEUEIMPL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_RUNQUEUEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryRessource.h"

#include <stack>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief File d'exécution pour accélérateur.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunQueueImpl
{
  friend class Arcane::Accelerator::Runner;
  friend class Arcane::Accelerator::RunQueue;
  friend class RunCommandImpl;
  friend class RunQueueImplStack;
  friend class RunnerImpl;

 private:

  RunQueueImpl(RunnerImpl* runner_impl, Int32 id, const RunQueueBuildInfo& bi);

 public:

  ~RunQueueImpl();

 public:

  RunQueueImpl(const RunQueueImpl&) = delete;
  RunQueueImpl(RunQueueImpl&&) = delete;
  RunQueueImpl& operator=(const RunQueueImpl&) = delete;
  RunQueueImpl& operator=(RunQueueImpl&&) = delete;

 public:

  static RunQueueImpl* create(RunnerImpl* r, const RunQueueBuildInfo& bi);
  static RunQueueImpl* create(RunnerImpl* r);

 public:

  eExecutionPolicy executionPolicy() const { return m_execution_policy; }
  RunnerImpl* runner() const { return m_runner_impl; }
  MemoryAllocationOptions allocationOptions() const;
  bool isAutoPrefetchCommand() const;
  IRunQueueStream* _internalStream() const { return m_queue_stream; }

 public:

  void addRef()
  {
    ++m_nb_ref;
  }
  void removeRef()
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
      _release();
  }

 private:

  RunCommandImpl* _internalCreateOrGetRunCommandImpl();
  IRunnerRuntime* _internalRuntime() const { return m_runtime; }
  void _internalFreeRunningCommands();
  void _internalBarrier();
  bool _isInPool() const { return m_is_in_pool; }
  void _release();
  void _setDefaultMemoryRessource();
  static RunQueueImpl* _reset(RunQueueImpl* p);

 private:

  RunnerImpl* m_runner_impl = nullptr;
  eExecutionPolicy m_execution_policy = eExecutionPolicy::None;
  IRunnerRuntime* m_runtime = nullptr;
  IRunQueueStream* m_queue_stream = nullptr;
  //! Pool de commandes
  std::stack<RunCommandImpl*> m_run_command_pool;
  //! Liste des commandes en cours d'exécution
  UniqueArray<RunCommandImpl*> m_active_run_command_list;
  //! Identifiant de la file
  Int32 m_id = 0;
  //! Indique si l'instance est dans un pool d'instance.
  bool m_is_in_pool = false;
  //! Nombre de références sur l'instance.
  std::atomic<Int32> m_nb_ref = 0;
  //! Indique si la file est asynchrone
  bool m_is_async = false;
  //! Ressource mémoire par défaut
  eMemoryRessource m_memory_ressource = eMemoryRessource::Unknown;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
