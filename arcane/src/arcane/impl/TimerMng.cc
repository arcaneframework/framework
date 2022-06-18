﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimerMng.cc                                                 (C) 2000-2022 */
/*                                                                           */
/* Implémentation d'un gestionnaire de timer.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/TimerMng.h"

#include "arcane/utils/Iterator.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/ArcaneException.h"
#include "arcane/Timer.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMainFactory.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimerMng::
TimerMng(ITraceMng* trace)
: TraceAccessor(trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimerMng::
beginTimer(Timer* timer)
{
  Real tvalue = _getRealTime();
  timer->_setStartTime(tvalue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
getTime(Timer* timer)
{
  Real tvalue = _getRealTime();
  Real active_value = tvalue - timer->_startTime();
  return active_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimerMng::
endTimer(Timer* timer)
{
  Real tvalue = _getRealTime();
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
_getRealTime()
{
  return platform::getRealTime();
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

