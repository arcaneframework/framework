// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Timer.cc                                                    (C) 2000-2025 */
/*                                                                           */
/* Gestion d'un timer.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Iostream.h"

#include "arcane/core/Timer.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ITimerMng.h"
#include "arcane/core/ITimeStats.h"

#include "arccore/trace/internal/TimeMetric.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::
Timer(ISubDomain* sd,const String& name,eTimerType type)
: Timer(sd->timerMng(),name,type)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::
Timer(ITimerMng* tm,const String& name,eTimerType type)
: m_timer_mng(tm)
, m_type(type)
, m_nb_activated(0)
, m_is_activated(false)
, m_activation_time(0.0)
, m_total_time(0.0)
, m_name(name)
, m_start_time(0.0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::
~Timer()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Timer::
start()
{
  if (m_is_activated)
    ARCANE_FATAL("Timer is already activated");
  
  //\todo Get time
  m_activation_time = 0.;
  m_timer_mng->beginTimer(this);

  ++m_nb_activated;
  m_is_activated = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real Timer::
stop()
{
  if (m_is_activated){
    m_activation_time = m_timer_mng->endTimer(this);
    m_is_activated = false;
    m_total_time += m_activation_time;
  }
  else
    ARCANE_FATAL("Timer is not activated");
  return m_activation_time;
}

void Timer::
reset()
{
  m_is_activated = false;
  m_activation_time = 0.0;
  m_total_time = 0.0;
 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Timer::Action::
_init()
{
  if (m_stats)
    if (!m_stats->isGathering())
      m_stats = 0;
  if (m_stats)
    m_stats->beginAction(m_action_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::Action::
Action(ISubDomain* sub_domain,const String& action_name,bool print_time)
: m_stats(0)
, m_action_name(action_name)
, m_print_time(print_time)
{
  if (sub_domain)
    m_stats = sub_domain->timeStats();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::Action::
Action(ITimeStats* stats,const String& action_name,bool print_time)
: m_stats(stats)
, m_action_name(action_name)
, m_print_time(print_time)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::Action::
~Action()
{
  if (m_stats)
    m_stats->endAction(m_action_name,m_print_time);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Timer::Phase::
_init()
{
  if (m_stats)
    if (!m_stats->isGathering())
      m_stats = 0;
  if (m_stats)
    m_stats->beginPhase(m_phase_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::Phase::
Phase(ISubDomain* sub_domain,eTimePhase pt)
: m_stats(0)
, m_phase_type(pt)
{
  if (sub_domain)
    m_stats = sub_domain->timeStats();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::Phase::
Phase(ITimeStats* stats,eTimePhase pt)
: m_stats(stats)
, m_phase_type(pt)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::Phase::
~Phase()
{
  if (m_stats)
    m_stats->endPhase(m_phase_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::SimplePrinter::
SimplePrinter(ITraceMng* tm,const String& msg)
: m_trace_mng(tm)
, m_begin_time(0.0)
, m_is_active(true)
, m_message(msg)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::SimplePrinter::
SimplePrinter(ITraceMng* tm,const String& msg,bool is_active)
: m_trace_mng(tm)
, m_begin_time(0.0)
, m_is_active(is_active)
, m_message(msg)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timer::SimplePrinter::
~SimplePrinter()
{
  if (m_is_active){
    Real end_time = platform::getRealTime();
    Real diff_time = end_time - m_begin_time;
    m_trace_mng->info() << m_message << "  time=" << diff_time;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Timer::SimplePrinter::
_init()
{
  if (m_is_active)
    m_begin_time = platform::getRealTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeMetricAction Timer::
phaseAction(ITimeStats* s,eTimePhase phase)
{
  ITimeMetricCollector* c = nullptr;
  if (s)
    c = s->metricCollector();
  if (!c)
    return TimeMetricAction();
  return TimeMetricAction(c,TimeMetricActionBuildInfo(String(),phase));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

