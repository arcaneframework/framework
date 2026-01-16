// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.h                                              (C) 2000-2026 */
/*                                                                           */
/* Implémentation d'une 'RunQueue'.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNQUEUEIMPL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNQUEUEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include "arccore/common/Array.h"

#include <stack>
#include <atomic>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief File d'exécution pour accélérateur.
 *
 * Cette classe gère l'implémentation d'une RunQueue.
 * La description des méthodes se trouve dans RunQueue.
 */
class ARCCORE_COMMON_EXPORT RunQueueImpl
{
  friend class Arcane::Accelerator::Runner;
  friend class Arcane::Accelerator::RunQueue;
  friend class RunCommandImpl;
  friend class RunQueueImplStack;
  friend class RunnerImpl;

  class Lock;

 private:

  RunQueueImpl(RunnerImpl* runner_impl, Int32 id, const RunQueueBuildInfo& bi);

 private:

  // Il faut utiliser _destroy() pour détruire l'instance.
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

  void copyMemory(const MemoryCopyArgs& args) const;
  void prefetchMemory(const MemoryPrefetchArgs& args) const;

  void recordEvent(RunQueueEvent& event);
  void waitEvent(RunQueueEvent& event);

  void setConcurrentCommandCreation(bool v);
  bool isConcurrentCommandCreation() const { return m_use_pool_mutex; }

  void dumpStats(std::ostream& ostr) const;
  bool isAsync() const { return m_is_async; }

  void _internalBarrier();
  IRunnerRuntime* _internalRuntime() const { return m_runtime; }
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
  void _internalFreeRunningCommands();
  bool _isInPool() const { return m_is_in_pool; }
  void _release();
  void _setDefaultMemoryRessource();
  void _addRunningCommand(RunCommandImpl* p);
  void _putInCommandPool(RunCommandImpl* p);
  void _freeCommandsInPool();
  void _checkPutCommandInPoolNoLock(RunCommandImpl* p);
  static RunQueueImpl* _reset(RunQueueImpl* p);
  static void _destroy(RunQueueImpl* q);

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
  eMemoryResource m_memory_ressource = eMemoryResource::Unknown;

  // Mutex pour les commandes (actif si \a m_use_pool_mutex est vrai)
  std::unique_ptr<std::mutex> m_pool_mutex;
  bool m_use_pool_mutex = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
