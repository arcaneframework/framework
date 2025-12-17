// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadBindingMng.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire pour punaiser les threads.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/ThreadBindingMng.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ConcurrencyUtils.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IProcessorAffinityService.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ThreadBindingMng::
ThreadBindingMng()
: m_thread_created_callback(new ObserverT<ThreadBindingMng>(this, &ThreadBindingMng::_createThreadCallback))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ThreadBindingMng::
~ThreadBindingMng()
{
  finalize();
  delete m_thread_created_callback;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ThreadBindingMng::
initialize(ITraceMng* tm,const String& strategy)
{
  m_trace_mng = tm;
  m_bind_strategy = strategy;
  // Si la strategie n'est pas nulle, s'attache à l'observable du TaskFactory
  // pour être notifié de la création du thread.
  if (!m_bind_strategy.null()){
    if (m_bind_strategy!="Simple")
      ARCANE_FATAL("Invalid strategy '{0}'. Valid values are : 'Simple'",m_bind_strategy);
    m_max_thread = TaskFactory::nbAllowedThread();
    if (tm)
      tm->info() << "Thread binding strategy is '" << m_bind_strategy << "'";
    TaskFactoryInternal::addThreadCreateObserver(m_thread_created_callback);
    m_has_callback = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ThreadBindingMng::
finalize()
{
  if (m_has_callback) {
    TaskFactoryInternal::removeThreadCreateObserver(m_thread_created_callback);
    m_has_callback = false;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ThreadBindingMng::
_createThreadCallback()
{
  // TODO: ne pas dépasser le nombre max de threads alloués, sinon afficher un
  // message d'avertissement
  ITraceMng* tm = m_trace_mng;
  Int32 thread_index = m_current_thread_index;
  ++m_current_thread_index;

  IProcessorAffinityService* pas = platform::getProcessorAffinityService();
  if (!pas){
    if (tm)
      tm->info() << "WARNING: Can not bind thread because there is no 'IProcessorAffinityService'";
    return;
  }

  if (tm){
    if (thread_index<m_max_thread){
      pas->bindThread(thread_index);
      tm->info() << "Binding thread index=" << thread_index << " cpuset=" << pas->cpuSetString();
    }
    else
      tm->info() << "WARNING: thread index is greater than maximum number of allowed thread. No binding done";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
