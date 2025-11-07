// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.h                                      (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'exécution d'une 'RunCommand'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNCOMMANDLAUNCHINFO_H
#define ARCANE_ACCELERATOR_CORE_RUNCOMMANDLAUNCHINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Profiling.h"
#include "arccore/base/ForLoopRunInfo.h"

#include "arcane/accelerator/core/KernelLaunchArgs.h"

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
class ARCANE_ACCELERATOR_CORE_EXPORT RunCommandLaunchInfo
{
  // Les fonctions suivantes permettent de lancer les kernels.
  template <typename SyclKernel, typename Lambda, typename LambdaArgs, typename... ReducerArgs>
  friend void _applyKernelSYCL(impl::RunCommandLaunchInfo& launch_info, SyclKernel kernel, Lambda& func,
                               const LambdaArgs& args, const ReducerArgs&... reducer_args);
  template <typename CudaKernel, typename Lambda, typename LambdaArgs, typename... RemainingArgs>
  friend void _applyKernelCUDA(impl::RunCommandLaunchInfo& launch_info, const CudaKernel& kernel, Lambda& func,
                               const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... other_args);
  template <typename HipKernel, typename Lambda, typename LambdaArgs, typename... RemainingArgs>
  friend void _applyKernelHIP(impl::RunCommandLaunchInfo& launch_info, const HipKernel& kernel, const Lambda& func,
                              const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... other_args);

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

  /*!
   * \brief Informations sur le nombre de block/thread/grille du noyau à lancer.
   *
   * Cette valeur n'est valide que pour si la commande est associée à un accélérateur.
   */
  KernelLaunchArgs kernelLaunchArgs() const { return m_kernel_launch_args; }

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
  eExecutionPolicy m_exec_policy = eExecutionPolicy::Sequential;
  KernelLaunchArgs m_kernel_launch_args;
  ForLoopRunInfo m_loop_run_info;
  Int64 m_total_loop_size = 0;
  impl::RunQueueImpl* m_queue_impl = nullptr;

 private:

  Int32 _sharedMemorySize() const;
  KernelLaunchArgs _threadBlockInfo(const void* func, Int32 shared_memory_size) const;
  NativeStream _internalNativeStream();
  void _doEndKernelLaunch();
  KernelLaunchArgs _computeKernelLaunchArgs() const;

 private:

  void _computeLoopRunInfo();

  // Pour SYCL: enregistre l'évènement associé à la dernière commande de la file
  // \a sycl_event_ptr est de type 'sycl::event*'.
  void _addSyclEvent(void* sycl_event_ptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
