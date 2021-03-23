// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.cc                                             (C) 2000-2021 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunQueueImpl.h"
#include "arcane/accelerator/Runner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl::
RunQueueImpl(Runner* runner,eExecutionPolicy exec_policy)
: m_runner(runner)
, m_execution_policy(exec_policy)
, m_runtime(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl::
~RunQueueImpl()
{
  while (!m_run_command_pool.empty()){
    RunCommand::_internalDestroyImpl(m_run_command_pool.top());
    m_run_command_pool.pop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_init()
{
  eExecutionPolicy p = executionPolicy();
  if (p==eExecutionPolicy::CUDA){
    m_runtime = getCUDARunQueueRuntime();
    if (!m_runtime)
      ARCANE_FATAL("Can not set execution policy to CUDA because no CUDARunQueueRuntime is available");
  }
  else
    m_runtime = getSequentialRunQueueRuntime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
release()
{
  m_runner->_internalFreeRunQueueImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImpl::
create(Runner* r,eExecutionPolicy exec_policy)
{
  return r->_internalCreateOrGetRunQueueImpl(exec_policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl* RunQueueImpl::
_internalCreateOrGetRunCommandImpl()
{
  auto& pool = m_run_command_pool;
  RunCommandImpl* p = nullptr;

  // TODO: rendre thread-safe
  if (!pool.empty()){
    p = pool.top();
    pool.pop();
  }
  else{
    p = RunCommand::_internalCreateImpl(this);
  }
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueImpl::
_internalFreeRunCommandImpl(RunCommandImpl* p)
{
  // TODO: rendre thread-safe
  m_run_command_pool.push(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
