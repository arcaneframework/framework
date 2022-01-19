// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.cc                                                   (C) 2000-2022 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/Runner.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/accelerator/core/RunQueueImpl.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"

#include <stack>
#include <map>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

namespace {
inline IRunQueueRuntime*
_getRuntime(eExecutionPolicy p)
{
  IRunQueueRuntime* runtime = nullptr;
  switch(p){
  case eExecutionPolicy::HIP:
    return impl::getHIPRunQueueRuntime();
  case eExecutionPolicy::CUDA:
    return impl::getCUDARunQueueRuntime();
  case eExecutionPolicy::Sequential:
    return impl::getSequentialRunQueueRuntime();
  case eExecutionPolicy::Thread:
    return impl::getThreadRunQueueRuntime();;
  }
  return runtime;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Runner::Impl
{
  class RunQueueImplStack
  {
   public:
    RunQueueImplStack(Runner* runner,eExecutionPolicy exec_policy,IRunQueueRuntime* runtime)
    : m_runner(runner), m_exec_policy(exec_policy), m_runtime(runtime){}
   public:
    bool empty() const { return m_stack.empty(); }
    void pop() { m_stack.pop(); }
    RunQueueImpl* top() { return m_stack.top(); }
    void push(RunQueueImpl* v) { m_stack.push(v); }
   public:
    RunQueueImpl* createRunQueue(const RunQueueBuildInfo& bi)
    {
      // Si pas de runtime, essaie de le récupérer. On le fait ici et aussi
      // lors de la création de l'instance car l'utilisateur a pu ajouter une
      // implémentation de runtime entre-temps (par exemple si l'instance de Runner
      // a été créée avant l'initialisation du runtime).
      if (!m_runtime)
        m_runtime = _getRuntime(m_exec_policy);
      if (!m_runtime)
        ARCANE_FATAL("Can not create RunQueue for execution policy '{0}' "
                     "because no RunQueueRuntime is available for this policy",m_exec_policy);
      Int32 x = ++m_nb_created;
      auto* q = new RunQueueImpl(m_runner,x,m_runtime,bi);
      q->m_is_in_pool = true;
      return q;
    }
   private:
    std::stack<RunQueueImpl*> m_stack;
    std::atomic<Int32> m_nb_created = -1;
    Runner* m_runner;
    eExecutionPolicy m_exec_policy;
    IRunQueueRuntime* m_runtime;
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
    _add(runner,eExecutionPolicy::HIP);
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
    IRunQueueRuntime* r = _getRuntime(exec_policy);
    auto* q = new RunQueueImplStack(runner,exec_policy,r);
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
  
  return pool->createRunQueue(RunQueueBuildInfo{});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueImpl* Runner::
_internalCreateOrGetRunQueueImpl(const RunQueueBuildInfo& bi)
{
  // Si on utilise les paramètres par défaut, on peut utilier une RunQueueImpl
  // issue du pool.
  eExecutionPolicy p = executionPolicy();
  if (bi.isDefault())
    return _internalCreateOrGetRunQueueImpl(p);
  IRunQueueRuntime* runtime = _getRuntime(p);
  ARCANE_CHECK_POINTER(runtime);
  auto* queue = new RunQueueImpl(this,0,runtime,bi);
  return queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Runner::
_internalFreeRunQueueImpl(RunQueueImpl* p)
{
  // TODO: rendre thread-safe
  if (p->_isInPool())
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

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
