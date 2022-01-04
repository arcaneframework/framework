// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EntryPoint.cc                                               (C) 2000-2019 */
/*                                                                           */
/* Point d'entrée d'un module.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/EntryPoint.h"
#include "arcane/IModule.h"
#include "arcane/IEntryPointMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/Timer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* IEntryPoint::WComputeLoop = "ComputeLoop";
const char* IEntryPoint::WBuild = "Build";
const char* IEntryPoint::WInit = "Init";
const char* IEntryPoint::WContinueInit = "ContinueInit";
const char* IEntryPoint::WStartInit = "StartInit";
const char* IEntryPoint::WRestore = "Restore";
const char* IEntryPoint::WOnMeshChanged = "OnMeshChanged";
const char* IEntryPoint::WOnMeshRefinement = "OnMeshRefinement";
const char* IEntryPoint::WExit = "Exit";

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntryPoint::
EntryPoint(IModule* module,const String& name,IFunctor* caller,
           const String& where,int property)
: m_sub_domain (module->subDomain())
, m_caller(caller)
, m_cpu_timer(new Timer(m_sub_domain,name,Timer::TimerVirtual))
, m_elapsed_timer(new Timer(m_sub_domain,name,Timer::TimerReal))
, m_name(name)
, m_module(module)
, m_where(where)
, m_property(property)
, m_nb_call(0)
, m_is_destroy_caller(false)
{
  // S'enregistre auprès du gestionnaire de points d'entrée.
  // qui s'occupera de la destruction.
  m_sub_domain->entryPointMng()->addEntryPoint(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntryPoint::
EntryPoint(const EntryPointBuildInfo& bi)
: m_sub_domain (bi.module()->subDomain())
, m_caller(bi.caller())
, m_cpu_timer(new Timer(m_sub_domain,bi.name(),Timer::TimerVirtual))
, m_elapsed_timer(new Timer(m_sub_domain,bi.name(),Timer::TimerReal))
, m_name(bi.name())
, m_module(bi.module())
, m_where(bi.where())
, m_property(bi.property())
, m_nb_call(0)
, m_is_destroy_caller(bi.isDestroyCaller())
{
  // S'enregistre auprès du gestionnaire de points d'entrée.
  // qui s'occupera de la destruction.
  m_sub_domain->entryPointMng()->addEntryPoint(this);

  {
    StringBuilder sb;
    sb.append(m_module->name());
    sb.append(".");
    sb.append(m_name);
    sb.append(".");
    sb.append(m_where);
    m_full_name = sb.toString();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntryPoint* EntryPoint::
create(const EntryPointBuildInfo& bi)
{
  EntryPoint* x = new EntryPoint(bi);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntryPoint::
~EntryPoint()
{
  delete m_cpu_timer;
  delete m_elapsed_timer;
  //FIXME ceci ne doit pas etre detruit si on passe par le wrapper C#
  if (m_is_destroy_caller)
    delete m_caller;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef __GNUG__
#pragma GCC optimize "O0"
#endif
void EntryPoint::
_getAddressForHyoda(void *next_entry_point_address)
{
#ifndef ARCANE_OS_WIN32
  if (platform::hasDotNETRuntime()) return;
  {
    char address[48+1];
    snprintf(address, sizeof(address)-1,"%p", next_entry_point_address);
    //m_sub_domain->traceMng()->debug()<< "\33[7m" << m_where <<"->"<< m_name << "\33[m" << " @"  << address;
  }
#endif
}
#ifdef __GNUG__
#pragma GCC reset_options
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void EntryPoint::
executeEntryPoint()
{
  // Les points d'entrée de la boucle de calcul ne
  // sont pas appelés si le module n'est pas actif.
  if (m_module->disabled() && m_where==WComputeLoop)
    return;

  // Dans le cas de GCC, on va piocher dans la module_vtable pour récupérer l'adresse à venir
  // que l'on va présenter à Hyoda
  if (!platform::hasDotNETRuntime()){
#ifdef __GNUG__
    _getAddressForHyoda((static_cast<IFunctorWithAddress*>(m_caller))->functorAddress());
#else
    _getAddressForHyoda();
#endif
  }

  Trace::Setter mclass(m_sub_domain->traceMng(),m_module->name());
  
  {
    Timer::Sentry ts_cpu(m_cpu_timer);
    Timer::Sentry ts_elapsed(m_elapsed_timer);
    Timer::Action ts_action1(m_sub_domain,m_module->name());
    Timer::Phase ts_phase(m_sub_domain,TP_Computation);
    Timer::Action ts_action2(m_sub_domain,m_name);
    m_caller->executeFunctor();
  }

  ++m_nb_call;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
lastTime() const
{
  return lastCPUTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
totalTime() const
{
  return totalCPUTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
lastCPUTime() const
{
  return m_cpu_timer->lastActivationTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
totalCPUTime() const
{
  return m_cpu_timer->totalTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
lastElapsedTime() const
{
  return m_elapsed_timer->lastActivationTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
totalElapsedTime() const
{
  return m_elapsed_timer->totalTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
totalTime(Timer::eTimerType type) const
{
  if (type==Timer::TimerVirtual)
    return totalCPUTime();
  return totalElapsedTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real EntryPoint::
lastTime(Timer::eTimerType type) const
{
  if (type==Timer::TimerVirtual)
    return lastCPUTime();
  return lastElapsedTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

