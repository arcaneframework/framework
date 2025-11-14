// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Gestion d'une commande sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunCommand.h"

#include "arcane/utils/ArraySimdPadder.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/NativeStream.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"
#include "arcane/accelerator/core/internal/ReduceMemoryImpl.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand::
RunCommand(const RunQueue& run_queue)
: m_p(run_queue._getCommandImpl())
{
  m_p->m_has_living_run_command = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand::
~RunCommand()
{
  m_p->m_has_living_run_command = false;
  m_p->_notifyDestroyRunCommand();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const TraceInfo& RunCommand::
traceInfo() const
{
  return m_p->traceInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& RunCommand::
kernelName() const
{
  return m_p->kernelName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 RunCommand::
nbThreadPerBlock() const
{
  return m_p->m_nb_thread_per_block;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addTraceInfo(const TraceInfo& ti)
{
  m_p->m_trace_info = ti;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addKernelName(const String& v)
{
  m_p->m_kernel_name = v;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addNbThreadPerBlock(Int32 v)
{
  if (v < 0)
    v = 0;
  if (v > 0 && v < 32)
    v = 32;
  m_p->m_nb_thread_per_block = v;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
setParallelLoopOptions(const ParallelLoopOptions& opt)
{
  m_p->m_parallel_loop_options = opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ParallelLoopOptions& RunCommand::
parallelLoopOptions() const
{
  return m_p->m_parallel_loop_options;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
RunCommand&
operator<<(RunCommand& command, const TraceInfo& trace_info)
{
  return command.addTraceInfo(trace_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::NativeStream RunCommand::
_internalNativeStream() const
{
  return m_p->internalStream()->nativeStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunQueueImpl* RunCommand::
_internalQueueImpl() const
{
  return m_p->m_queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunCommandImpl* RunCommand::
_internalCreateImpl(impl::RunQueueImpl* queue)
{
  return new impl::RunCommandImpl(queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalDestroyImpl(impl::RunCommandImpl* p)
{
  delete p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_allocateReduceMemory(Int32 nb_grid)
{
  auto& mem_list = m_p->m_active_reduce_memory_list;
  if (!mem_list.empty()) {
    for (auto& x : mem_list)
      x->setGridSizeAndAllocate(nb_grid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalNotifyBeginLaunchKernel()
{
  m_p->notifyBeginLaunchKernel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalNotifyEndLaunchKernel()
{
  m_p->notifyEndLaunchKernel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalNotifyBeginLaunchKernelSyclEvent(void* sycl_event_ptr)
{
  m_p->notifyLaunchKernelSyclEvent(sycl_event_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ForLoopOneExecStat* RunCommand::
_internalCommandExecStat()
{
  return m_p->m_loop_one_exec_stat_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 RunCommand::
_addSharedMemory(Int32 size)
{
  Int32 current_size = m_p->m_shared_memory_size;
  m_p->m_shared_memory_size += ArraySimdPadder::getSizeWithSpecificPadding<16>(size);
  return current_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 RunCommand::
_sharedMemory() const
{
  return m_p->m_shared_memory_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
