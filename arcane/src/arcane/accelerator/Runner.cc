// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.cc                                                   (C) 2000-2021 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Runner.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/Concurrency.h"

#include "arcane/accelerator/RunQueueImpl.h"

#include <stack>
#include <map>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

class Runner::Impl
{
  class RunQueueImplStack
  {
   public:
    RunQueueImplStack(Runner* runner,eExecutionPolicy exec_policy)
    : m_runner(runner), m_exec_policy(exec_policy){}
   public:
    bool empty() const { return m_stack.empty(); }
    void pop() { m_stack.pop(); }
    RunQueueImpl* top() { return m_stack.top(); }
    void push(RunQueueImpl* v) { m_stack.push(v); }
   public:
    RunQueueImpl* createRunQueue()
    {
      Int32 x = ++m_nb_created;
      return new RunQueueImpl(m_runner,m_exec_policy,x);
    }
   private:
    std::stack<RunQueueImpl*> m_stack;
    std::atomic<Int32> m_nb_created = -1;
    Runner* m_runner;
    eExecutionPolicy m_exec_policy;
  };

 public:
  ~Impl()
  {
    for( auto& x : m_run_queue_pool_map ){
      _freePool(x.second);
      delete x.second;
    }
  }
 public:
  void build(Runner* runner)
  {
    _add(runner,eExecutionPolicy::Sequential);
    _add(runner,eExecutionPolicy::Thread);
    _add(runner,eExecutionPolicy::CUDA);
  }
 public:
  RunQueueImplStack* getPool(eExecutionPolicy exec_policy)
  {
    auto x = m_run_queue_pool_map.find(exec_policy);
    if (x==m_run_queue_pool_map.end())
      ARCANE_FATAL("No RunQueueImplStack for execution policy '{0}'",(int)exec_policy);
    return x->second;
  }
 public:
  eExecutionPolicy m_execution_policy = eExecutionPolicy::Sequential;
 private:
  std::map<eExecutionPolicy,RunQueueImplStack*> m_run_queue_pool_map;
 private:
  void _freePool(RunQueueImplStack* s)
  {
    while (!s->empty()){
      delete s->top();
      s->pop();
    }
  }
  void _add(Runner* runner,eExecutionPolicy exec_policy)
  {
    auto* q = new RunQueueImplStack(runner,exec_policy);
    m_run_queue_pool_map.insert(std::make_pair(exec_policy,q));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
Runner()
: m_p(new Impl())
{
  m_p->build(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner::
~Runner()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* Runner::
_internalCreateOrGetRunQueueImpl(eExecutionPolicy exec_policy)
{
  auto pool = m_p->getPool(exec_policy);

  // TODO: rendre thread-safe
  if (!pool->empty()){
    RunQueueImpl* p = pool->top();
    pool->pop();
    return p;
  }
  
  return pool->createRunQueue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
_internalFreeRunQueueImpl(RunQueueImpl* p)
{
  // TODO: rendre thread-safe
  m_p->getPool(p->executionPolicy())->push(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eExecutionPolicy Runner::
executionPolicy() const
{
  return m_p->m_execution_policy;
}

void Runner::
setExecutionPolicy(eExecutionPolicy v)
{
  m_p->m_execution_policy = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_EXPORT void
initializeRunner(Runner& runner,ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  String accelerator_runtime = acc_info.acceleratorRuntime();
  tm->info() << "AcceleratorRuntime=" << accelerator_runtime;
  if (accelerator_runtime=="cuda"){
    tm->info() << "Using CUDA runtime";
    runner.setExecutionPolicy(eExecutionPolicy::CUDA);
  }
  else if (TaskFactory::isActive()){
    tm->info() << "Using Task runtime";
    runner.setExecutionPolicy(eExecutionPolicy::Thread);
  }
  else{
    tm->info() << "Using Sequential runtime";
    runner.setExecutionPolicy(eExecutionPolicy::Sequential);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
