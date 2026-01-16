// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.h                                      (C) 2000-2026 */
/*                                                                           */
/* Informations pour l'exécution d'une 'RunCommand'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNCOMMANDLAUNCHINFO_H
#define ARCCORE_COMMON_ACCELERATOR_RUNCOMMANDLAUNCHINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Profiling.h"
#include "arccore/base/ForLoopRunInfo.h"

#include "arccore/common/accelerator/KernelLaunchArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Object temporaire pour conserver les informations d'exécution d'une
 * commande et regrouper les tests.
 */
class ARCCORE_COMMON_EXPORT RunCommandLaunchInfo
{
  // Les classes suivantes permettent de lancer les kernels.
  friend class CudaKernelLauncher;
  friend class HipKernelLauncher;
  friend class SyclKernelLauncher;

 public:

  using ThreadBlockInfo = KernelLaunchArgs;

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

  //! Calcule et retourne les informations pour les boucles multi-thread
  ParallelLoopOptions computeParallelLoopOptions() const;

  /*!
   * \brief Informations d'exécution de la boucle.
   *
   * Ces informations ne sont valides que si executionPolicy()==eExecutionPolicy::Thread
   * et si beginExecute() a été appelé.
   */
  const ForLoopRunInfo& loopRunInfo() const { return m_loop_run_info; }

  //! Taille totale de la boucle
  Int64 totalLoopSize() const { return m_total_loop_size; }

 private:

  RunCommand& m_command;
  bool m_has_exec_begun = false;
  bool m_is_notify_end_kernel_done = false;
  bool m_is_need_barrier = false;
  eExecutionPolicy m_exec_policy = eExecutionPolicy::Sequential;
  KernelLaunchArgs m_kernel_launch_args;
  ForLoopRunInfo m_loop_run_info;
  Int64 m_total_loop_size = 0;
  RunQueueImpl* m_queue_impl = nullptr;

 private:

  //! Calcule les arguments pour lancer le noyau dont l'adresse est \a func
  KernelLaunchArgs _computeKernelLaunchArgs(const void* func) const;
  NativeStream _internalNativeStream();
  void _doEndKernelLaunch();
  void _computeInitialKernelLaunchArgs();

  // Pour test uniquement avec CUDA
  bool _isUseCooperativeLaunch() const;
  bool _isUseCudaLaunchKernel() const;
  void _setIsNeedBarrier(bool v);

 private:

  void _computeLoopRunInfo();

  // Pour SYCL: enregistre l'évènement associé à la dernière commande de la file
  // \a sycl_event_ptr est de type 'sycl::event*'.
  void _addSyclEvent(void* sycl_event_ptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
