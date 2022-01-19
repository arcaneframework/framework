// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueImpl.cc                                             (C) 2000-2022 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunQueueImpl.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/IRunQueueRuntime.h"
#include "arcane/accelerator/core/IRunQueueStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl::
RunQueueImpl(Runner* runner,Int32 id,IRunQueueRuntime* runtime,const RunQueueBuildInfo& bi)
: m_runner(runner)
, m_execution_policy(runtime->executionPolicy())
, m_runtime(runtime)
, m_queue_stream(runtime->createStream(bi))
, m_id(id)
{
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
  delete m_queue_stream;
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
create(Runner* r)
{
  return r->_internalCreateOrGetRunQueueImpl(r->executionPolicy());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* RunQueueImpl::
create(Runner* r,const RunQueueBuildInfo& bi)
{
  return r->_internalCreateOrGetRunQueueImpl(bi);
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
