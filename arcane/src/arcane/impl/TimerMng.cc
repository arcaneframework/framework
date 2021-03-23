// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimerMng.cc                                                 (C) 2000-2019 */
/*                                                                           */
/* Implémentation d'un gestionnaire de timer.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iterator.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/ArcaneException.h"
#include "arcane/Timer.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMainFactory.h"

#include "arcane/impl/TimerMng.h"

#ifdef ARCANE_OS_WIN32
#include <windows.h>
#define ARCANE_TIMER_USE_CLOCK
#else
#include <sys/time.h>
#endif
#include <time.h>
#include <errno.h>

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_TIMER_USE_CLOCK
static clock_t current_clock_value = 0;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimerMng::
TimerMng(ITraceMng* trace)
: TraceAccessor(trace)
, m_nb_timer(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimerMng::~TimerMng() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimerMng::
beginTimer(Timer* timer)
{
  if (timer->type()==Timer::TimerVirtual){
    if (m_nb_timer.fetch_add(1)==0)
      _setVirtualTime();
  }
  Real tvalue = _getTime(timer);
  timer->_setStartTime(tvalue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
getTime(Timer* timer)
{
  Real tvalue = _getTime(timer);
  Real active_value = tvalue - timer->_startTime();
  return active_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
endTimer(Timer* timer)
{
  Real tvalue = _getTime(timer);
  if (timer->type()==Timer::TimerVirtual)
    --m_nb_timer;
  Real active_value = tvalue - timer->_startTime();
  return active_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TimerMng::
hasTimer(Timer* timer)
{
  return timer->timerMng()==this && timer->isActivated();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
_getTime(Timer* t)
{
  switch(t->type()){
   case Timer::TimerVirtual:
     return _getVirtualTime();
   case Timer::TimerReal:
     return _getRealTime();
  }
  return 0.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimerMng::
_errorInTimer(const String& msg,int retcode)
{
  warning() << "In TimerMng::_errorInTimer() "
            << msg << " return code: " << retcode << " errno: " << errno;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimerMng::
_setVirtualTime()
{
#ifdef ARCANE_TIMER_USE_CLOCK
	current_clock_value = ::clock();
#else
	struct itimerval time_val;
  struct itimerval otime_val;
  time_val.it_value.tv_sec = 5000000;
  time_val.it_value.tv_usec = 0;
  time_val.it_interval.tv_sec = 0;
  time_val.it_interval.tv_usec = 0;
  int r = ::setitimer(ITIMER_VIRTUAL,&time_val,&otime_val);
  if (r!=0)
    _errorInTimer("setitimer()",r);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
_getVirtualTime()
{
#ifdef ARCANE_TIMER_USE_CLOCK
	clock_t cv = ::clock();
	Real diffv = static_cast<Real>(cv-current_clock_value);
	return diffv / CLOCKS_PER_SEC;
#else
	struct itimerval time_val;
  int r = ::getitimer(ITIMER_VIRTUAL,&time_val);
  if (r!=0)
    _errorInTimer("getitimer()",r);
  double v = ((double)time_val.it_value.tv_sec) * 1.0
    + ((double)time_val.it_value.tv_usec) * 1e-6;
  return (5000000. - v);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimerMng::
_setRealTime()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
_getRealTime()
{
  return platform::getRealTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

