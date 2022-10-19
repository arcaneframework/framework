// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchInfo.cc                                     (C) 2000-2022 */
/*                                                                           */
/* Informations pour l'exécution d'une 'RunCommand'.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunQueueInternal.h"
#include "arcane/accelerator/RunQueue.h"
#include "arcane/accelerator/IRunQueueStream.h"

#include "arcane/utils/CheckedConvert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
RunCommandLaunchInfo(RunCommand& command)
: m_command(command)
{
  _begin();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
RunCommandLaunchInfo(RunCommand& command,Int64 total_loop_size)
: m_command(command)
{
  m_thread_block_info = computeThreadBlockInfo(total_loop_size);
  _begin();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandLaunchInfo::
~RunCommandLaunchInfo()
{
  // Normalement ce test est toujours faux sauf s'il y a eu une exception
  // pendant le lancement du noyau de calcul.
  if (!m_is_notify_end_kernel_done)
    m_queue_stream->notifyEndKernel(m_command);
  m_command._resetInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
_begin()
{
  RunQueue& queue = m_command._internalQueue();
  m_exec_policy = queue.executionPolicy();
  m_queue_stream = queue._internalStream();
  m_runtime = queue._internalRuntime();
  m_queue_stream->notifyBeginKernel(m_command);
  m_command._allocateReduceMemory(m_thread_block_info.nb_block_per_grid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
beginExecute()
{
  if (m_has_exec_begun)
    ARCANE_FATAL("beginExecute() has to be called before endExecute()");
  m_has_exec_begun = true;
  m_begin_time = platform::getRealTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
endExecute()
{
  if (!m_has_exec_begun)
    ARCANE_FATAL("beginExecute() has to be called before endExecute()");
  m_is_notify_end_kernel_done = true;
  m_queue_stream->notifyEndKernel(m_command);
  RunQueue& queue = m_command._internalQueue();
  if (!queue.isAsync())
    m_queue_stream->barrier();
  double end_time = platform::getRealTime();
  double diff_time = end_time - m_begin_time;
  queue._addCommandTime(diff_time);

  // Statistiques d'exécution si demandé
  ForLoopOneExecStat* exec_info = m_loop_run_info.execStat();
  if (exec_info){
    Int64 v_as_int64 = static_cast<Int64>(diff_time * 1.0e9);
    exec_info->setExecTime(v_as_int64);
    //std::cout << "END_EXEC exec_info=" << m_loop_run_info.traceInfo().traceInfo() << "\n";
    ProfilingRegistry::threadLocalInstance()->merge(*exec_info,m_loop_run_info.traceInfo());
  }
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

auto RunCommandLaunchInfo::
computeThreadBlockInfo(Int64 full_size) const -> ThreadBlockInfo
{
  int threads_per_block = m_command.nbThreadPerBlock();
  if (threads_per_block<=0)
    threads_per_block = 256;
  Int64 big_b = (full_size + threads_per_block - 1) / threads_per_block;
  int blocks_per_grid = CheckedConvert::toInt32(big_b);
  return { blocks_per_grid, threads_per_block };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelLoopOptions RunCommandLaunchInfo::
computeParallelLoopOptions(Int64 full_size) const
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
    Int32 grain_size = static_cast<Int32>((double)full_size / (nb_thread * 10.0));
    opt.setGrainSize(grain_size);
  }
  return opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandLaunchInfo::
computeLoopRunInfo(Int64 full_size)
{
  if (m_has_exec_begun)
    ARCANE_FATAL("computeLoopRunInfo() has to be called before beginExecute()");
  ForLoopTraceInfo lti(m_command.traceInfo(), m_command.kernelName());
  m_loop_run_info = ForLoopRunInfo(computeParallelLoopOptions(full_size), lti);

  // TODO: ne faire que si les traces sont actives
  if (TaskFactory::executionStatLevel()>0) {
    m_loop_one_exec_stat.reset();
    m_loop_run_info.setExecStat(&m_loop_one_exec_stat);
  }
  else
    m_loop_run_info.setExecStat(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
