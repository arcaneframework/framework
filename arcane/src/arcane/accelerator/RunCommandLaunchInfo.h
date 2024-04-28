// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.h                                      (C) 2000-2024 */
/*                                                                           */
/* Informations pour l'exécution d'une 'RunCommand'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLAUNCHINFO_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLAUNCHINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ConcurrencyUtils.h"
#include "arcane/utils/Profiling.h"

#include "arcane/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Object temporaire pour conserver les informations d'exécution d'une
 * commande et regrouper les tests.
 */
class ARCANE_ACCELERATOR_EXPORT RunCommandLaunchInfo
{
 public:

  struct ThreadBlockInfo
  {
    int nb_block_per_grid = 0;
    int nb_thread_per_block = 0;
  };

 public:

  RunCommandLaunchInfo(RunCommand& command, Int64 total_loop_size);
  ~RunCommandLaunchInfo();
  RunCommandLaunchInfo(const RunCommandLaunchInfo&) = delete;
  RunCommandLaunchInfo operator=(const RunCommandLaunchInfo&) = delete;

 public:

  eExecutionPolicy executionPolicy() const { return m_exec_policy; }

  /*!
   * \brief Indique qu'on commence l'exécution de la commande.
   *
   * Doit toujours être appelé avant de lancer la commande pour être
   * sur que cette méthode est appelée en cas d'exception.
   */
  void beginExecute();

  /*!
   * \brief Signale la fin de l'exécution.
   *
   * Si la file associée à la commande est asynchrone, la commande
   * peut continuer à s'exécuter après cet appel.
   */
  void endExecute();

  //! Informations sur le nombre de block/thread/grille du noyau à lancer.
  ThreadBlockInfo threadBlockInfo() const { return m_thread_block_info; }

  //! Calcul les informations pour les boucles multi-thread
  ParallelLoopOptions computeParallelLoopOptions() const;

  //! Calcule la valeur de loopRunInfo()
  void computeLoopRunInfo();

  //! Informations d'exécution de la boucle
  const ForLoopRunInfo& loopRunInfo() const { return m_loop_run_info; }

  //! Taille totale de la boucle
  Int64 totalLoopSize() const { return m_total_loop_size; }

 public:

  void* _internalStreamImpl();

 private:

  RunCommand& m_command;
  bool m_has_exec_begun = false;
  bool m_is_notify_end_kernel_done = false;
  IRunnerRuntime* m_runtime = nullptr;
  IRunQueueStream* m_queue_stream = nullptr;
  eExecutionPolicy m_exec_policy = eExecutionPolicy::Sequential;
  ThreadBlockInfo m_thread_block_info;
  ForLoopRunInfo m_loop_run_info;
  Int64 m_total_loop_size = 0;

 private:

  void _begin();
  void _doEndKernelLaunch();
  ThreadBlockInfo _computeThreadBlockInfo() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
