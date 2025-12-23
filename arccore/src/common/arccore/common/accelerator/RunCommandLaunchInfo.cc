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

#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "KernelLaunchArgs.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/CheckedConvert.h"
#include "arccore/base/ConcurrencyBase.h"

#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/common/accelerator/NativeStream.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
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
    _computeInitialKernelLaunchArgs();
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
    ARCCORE_FATAL("beginExecute() has already been called");
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
    ARCCORE_FATAL("beginExecute() has to be called before endExecute()");
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
  if (!q->isAsync() || m_is_need_barrier)
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
 * \brief Calcule la valeur initiale de block/thread/grille du noyau
 * en fonction de \a full_size.
 */
void RunCommandLaunchInfo::
_computeInitialKernelLaunchArgs()
{
  int threads_per_block = m_command.nbThreadPerBlock();
  if (threads_per_block<=0)
    threads_per_block = 256;
  Int64 big_b = (m_total_loop_size + threads_per_block - 1) / threads_per_block;
  int blocks_per_grid = CheckedConvert::toInt32(big_b);
  m_kernel_launch_args = KernelLaunchArgs(blocks_per_grid, threads_per_block, m_command._sharedMemory());
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
_computeKernelLaunchArgs(const void* func) const
{
  impl::IRunnerRuntime* r = m_queue_impl->_internalRuntime();

  return r->computeKernalLaunchArgs(m_kernel_launch_args, func,
                                    totalLoopSize());
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

bool RunCommandLaunchInfo::
_isUseCooperativeLaunch() const
{
  // Indique si on utilise cudaLaunchCooperativeKernel()
  return false;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunCommandLaunchInfo::
_isUseCudaLaunchKernel() const
{
  // Indique si on utilise cudaLaunchKernel() au lieu de kernel<<<...>>>.
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_setIsNeedBarrier(bool v)
{
  m_is_need_barrier = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
