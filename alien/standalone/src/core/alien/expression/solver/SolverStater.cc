// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* alienc                                         (C) 2000-2024              */
/*                                                                           */
/* Interface C for alien                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "SolverStater.h"

#ifdef WIN32
#include <windows.h>
#define ARCANE_TIMER_USE_CLOCK
#else
#include <errno.h>
#include <sys/time.h>
#endif

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/TraceInfo.h>
#include <ctime>

/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_TIMER_USE_CLOCK
static clock_t current_clock_value = 0;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real BaseSolverStater::_getVirtualTime()
{
  // From Arcane 1.16.3 to work with historical timers and Windows
#ifdef ARCANE_TIMER_USE_CLOCK
  clock_t cv = ::clock();
  Real diffv = static_cast<Real>(cv - current_clock_value);
  return diffv / CLOCKS_PER_SEC;
#else
  struct itimerval time_val;
  int r = ::getitimer(ITIMER_VIRTUAL, &time_val);
  if (r != 0)
    _errorInTimer("getitimer()", r);
  Real v = static_cast<Real>(time_val.it_value.tv_sec) * 1. + static_cast<Real>(time_val.it_value.tv_usec) * 1e-6;
  return (5000000. - v);
#endif
}

/*---------------------------------------------------------------------------*/

Real BaseSolverStater::_getRealTime()
{
  // From Arcane 1.16.3 to work with old timers and Windows.
#ifdef WIN32
  SYSTEMTIME t;
  GetSystemTime(&t);
  Real hour = t.wHour * 3600.0;
  Real minute = t.wMinute * 60.0;
  Real second = t.wSecond;
  Real milli_second = t.wMilliseconds * 1e-3;
  return (hour + minute + second + milli_second);
#else
  struct timeval tp;
  int r = gettimeofday(&tp, 0);
  if (r != 0)
    _errorInTimer("gettimeofday()", r);
  Real tvalue =
  (static_cast<Real>(tp.tv_sec) * 1. + static_cast<Real>(tp.tv_usec) * 1.e-6);
  return tvalue;
#endif
}

/*---------------------------------------------------------------------------*/

void BaseSolverStater::_errorInTimer(const String& msg, int retcode)
{
  throw FatalErrorException(
  A_FUNCINFO, String::format("{0} return code: {1} errno: {2}", msg, retcode, errno));
}

/*---------------------------------------------------------------------------*/

void BaseSolverStater::_startTimer()
{
  ALIEN_ASSERT((m_state == eNone), ("Unexpected SolverStater state %d", m_state));
  m_real_time = _getRealTime();
  m_cpu_time = _getVirtualTime();
}

/*---------------------------------------------------------------------------*/

void BaseSolverStater::_stopTimer()
{
  ALIEN_ASSERT((m_state != eNone), ("Unexpected SolverStater state %d", m_state));
  m_real_time = _getRealTime() - m_real_time;
  m_cpu_time = _getVirtualTime() - m_cpu_time;
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
