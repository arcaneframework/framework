// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'exécution d'une 'RunCommand'.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunCommandLaunchInfo.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/CheckedConvert.h"
#include "arccore/base/ConcurrencyBase.h"

#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"
#include "arcane/accelerator/core/NativeStream.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
RunCommandLaunchInfo(RunCommand& command, Int64 total_loop_size)
: m_command(command)
, m_total_loop_size(total_loop_size)
{
  m_queue_impl = m_command._internalQueueImpl();
  m_exec_policy = m_queue_impl->executionPolicy();

  // Le calcul des informations de kernel n'est utile que sur accélérateur
  if (isAcceleratorPolicy(m_exec_policy)) {
    m_kernel_launch_args = _computeKernelLaunchArgs();
    m_command._allocateReduceMemory(m_kernel_launch_args.nbBlockPerGrid());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
~RunCommandLaunchInfo()
{
  // Notifie de la fin de lancement du noyau. Normalement, cela est déjà fait
  // sauf s'il y a eu une exception pendant le lancement du noyau de calcul.
  _doEndKernelLaunch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
beginExecute()
{
  if (m_has_exec_begun)
    ARCANE_FATAL("beginExecute() has already been called");
  m_has_exec_begun = true;
  m_command._internalNotifyBeginLaunchKernel();
  if (m_exec_policy == eExecutionPolicy::Thread)
    _computeLoopRunInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notifie de la fin de lancement de la commande.
 *
 * A noter que si la commande est asynchrone, son exécution peut continuer
 * après l'appel à cette méthode.
 */
void RunCommandLaunchInfo::
endExecute()
{
  if (!m_has_exec_begun)
    ARCANE_FATAL("beginExecute() has to be called before endExecute()");
  _doEndKernelLaunch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_doEndKernelLaunch()
{
  if (m_is_notify_end_kernel_done)
    return;
  m_is_notify_end_kernel_done = true;
  m_command._internalNotifyEndLaunchKernel();

  impl::RunQueueImpl* q = m_queue_impl;
  if (!q->isAsync())
    q->_internalBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NativeStream RunCommandLaunchInfo::
_internalNativeStream()
{
  return m_command._internalNativeStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule le nombre de block/thread/grille du noyau en fonction de \a full_size.
 */
KernelLaunchArgs RunCommandLaunchInfo::
_computeKernelLaunchArgs() const
{
  int threads_per_block = m_command.nbThreadPerBlock();
  if (threads_per_block<=0)
    threads_per_block = 256;
  Int64 big_b = (m_total_loop_size + threads_per_block - 1) / threads_per_block;
  int blocks_per_grid = CheckedConvert::toInt32(big_b);
  return { blocks_per_grid, threads_per_block };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelLoopOptions RunCommandLaunchInfo::
computeParallelLoopOptions() const
{
  ParallelLoopOptions opt = m_command.parallelLoopOptions();
  const bool use_dynamic_compute = true;
  // Calcule une taille de grain par défaut si cela n'est pas renseigné dans
  // les options. Par défaut on fait en sorte de faire un nombre d'itérations
  // égale à 2 fois le nombre de threads utilisés.
  if (use_dynamic_compute && opt.grainSize() == 0) {
    Int32 nb_thread = opt.maxThread();
    if (nb_thread <= 0)
      nb_thread = ConcurrencyBase::maxAllowedThread();
    if (nb_thread <= 0)
      nb_thread = 1;
    Int32 grain_size = static_cast<Int32>((double)m_total_loop_size / (nb_thread * 2.0));
    opt.setGrainSize(grain_size);
  }
  return opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule la valeur de m_loop_run_info.
 *
 * Cela n'est utile qu'en mode multi-thread.
 */
void RunCommandLaunchInfo::
_computeLoopRunInfo()
{
  ForLoopTraceInfo lti(m_command.traceInfo(), m_command.kernelName());
  m_loop_run_info = ForLoopRunInfo(computeParallelLoopOptions(), lti);
  m_loop_run_info.setExecStat(m_command._internalCommandExecStat());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine la configuration du kernel.
 *
 * La configuration est dépendante du runtime sous-jacent. Pour CUDA et ROCM,
 * il s'agit d'un nombre de blocs et de thread.
 *
 * Il est possible de calculer dynamiquement les valeurs optimales pour
 * maximiser l'occupation.
 */
KernelLaunchArgs RunCommandLaunchInfo::
_threadBlockInfo(const void* func, Int32 shared_memory_size) const
{
  return m_queue_impl->_internalRuntime()->computeKernalLaunchArgs(m_kernel_launch_args, func, totalLoopSize(), shared_memory_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_addSyclEvent(void* sycl_event_ptr)
{
  m_command._internalNotifyBeginLaunchKernelSyclEvent(sycl_event_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 RunCommandLaunchInfo::
_sharedMemorySize() const
{
  return m_command._sharedMemory();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
