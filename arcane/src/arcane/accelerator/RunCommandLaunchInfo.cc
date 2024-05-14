﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Informations pour l'exécution d'une 'RunCommand'.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunCommandLaunchInfo.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"

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
  m_thread_block_info = _computeThreadBlockInfo();
  _begin();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
~RunCommandLaunchInfo()
{
  // Notifie de la fin de lancement du noyau. Normalement cela est déjà fait
  // sauf s'il y a eu une exception pendant le lancement du noyau de calcul.
  _doEndKernelLaunch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_begin()
{
  const RunQueue& queue = m_command._internalQueue();
  m_exec_policy = queue.executionPolicy();
  m_queue_stream = queue._internalStream();
  m_runtime = queue._internalRuntime();
  m_command._allocateReduceMemory(m_thread_block_info.nb_block_per_grid);
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

  const RunQueue& q = m_command._internalQueue();
  if (!q.isAsync())
    q.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* RunCommandLaunchInfo::
_internalStreamImpl()
{
  return m_queue_stream->_internalImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Calcule le nombre de block/thread/grille du noyau en fonction de \a full_size
auto RunCommandLaunchInfo::
_computeThreadBlockInfo() const -> ThreadBlockInfo
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
  const bool use_dynamic_compute = false;
  // Calcule une taille de grain par défaut si cela n'est pas renseigné dans
  // les options
  if (use_dynamic_compute && opt.grainSize() == 0) {
    Int32 nb_thread = opt.maxThread();
    if (nb_thread <= 0)
      nb_thread = TaskFactory::nbAllowedThread();
    if (nb_thread <= 0)
      nb_thread = 1;
    Int32 grain_size = static_cast<Int32>((double)m_total_loop_size / (nb_thread * 10.0));
    opt.setGrainSize(grain_size);
  }
  return opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
computeLoopRunInfo()
{
  if (m_has_exec_begun)
    ARCANE_FATAL("computeLoopRunInfo() has to be called before beginExecute()");
  ForLoopTraceInfo lti(m_command.traceInfo(), m_command.kernelName());
  m_loop_run_info = ForLoopRunInfo(computeParallelLoopOptions(), lti);
  m_loop_run_info.setExecStat(m_command._internalCommandExecStat());
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

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
