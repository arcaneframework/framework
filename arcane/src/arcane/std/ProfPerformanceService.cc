// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfPerformanceService.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Performance information using profiling signals.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"

#include "arcane/utils/IProfilingService.h"

#include "arcane/std/ProfilingInfo.h"

// NOTE: Since this file requires libunwind, it is only compiled on
// UNIX-style OS.

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdio.h>
#include <cxxabi.h>

#include <sys/time.h>
#include <signal.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Profiling service using 'setitimer'.
 */
class ProfPerformanceService
: public AbstractService
, public IProfilingService
{
 public:

  explicit ProfPerformanceService(const ServiceBuildInfo& sbi);
  ~ProfPerformanceService() override;

 public:

  void initialize() override;
  bool isInitialized() const override { return m_is_initialized; }
  void startProfiling() override;
  void switchEvent() override;
  void stopProfiling() override;
  void printInfos(bool dump_file) override;
  void getInfos(Int64Array&) override;
  void dumpJSON(JSONWriter& writer) override;
  void reset() override;
  ITimerMng* timerMng() override { return nullptr; }

 public:

  bool m_is_initialized = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(ProfPerformanceService,
                                    IProfilingService,
                                    ProfProfilingService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfPerformanceService::
ProfPerformanceService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfPerformanceService::
~ProfPerformanceService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static void
_setTimer(Integer usecond)
{
  struct itimerval time_val;
  struct itimerval otime_val;
  time_val.it_value.tv_sec     = 0;
  time_val.it_value.tv_usec    = usecond;
  time_val.it_interval.tv_sec  = 0;
  time_val.it_interval.tv_usec = 0;
  int r = setitimer(ITIMER_PROF,&time_val,&otime_val);
  if (r!=0)
    cout << "** ERROR in setitimer r=" << r << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
int nb_total = 0;

ProfInfos* global_infos = nullptr;
bool global_is_active = false;
//! Timer time in microseconds
int global_timer_period = 10000;
}

extern "C" void
arcane_prof_handler()
{
  static bool is_in_handler = false;
  // On Linux with gcc, exceptions use libunwind contained
  // in gcc, and this can cause deadlocks with our usage
  // if this handler is called during exception unwinding.
  // To avoid this problem, we do nothing as long as an exception is
  // active.
  if (Exception::hasPendingException()){
    cout << "** WARNING: ProfHandler in pending exception\n";
    return;
  }
  if (is_in_handler){
    cout << "** In handler\n";
    return;
  }
  is_in_handler = true;
  ++nb_total;

  int overflow_event[MAX_COUNTER];
  int nb_overflow_event = 1;
  overflow_event[0] = 0;
  
  unw_word_t func_ip = 0;
  unw_word_t offset = 0;
  {
    unw_cursor_t cursor;
    unw_context_t uc;
    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    int current_func = 0;
    String message;

    // 3 indicates the number of functions before the correct one
    // (there is this function, then _arcaneProfilingSigFunc
    // and the signal itself.
    while (unw_step(&cursor) > 0 && current_func<3) {
      //while (current_func<3) {
      //unw_step(&cursor);
      unw_get_reg(&cursor, UNW_REG_IP, &func_ip);
#if 0
      char func_name_buf[10000];
      unw_get_proc_name(&cursor,func_name_buf,10000,&offset);
      cout << "** I=" << current_func << " FUNC NAME=" << func_name_buf
           << " ip=" << (void*)func_ip << '\n';
#endif
      ++current_func;
    }
  }

  global_infos->addEvent((void*)(func_ip+offset),overflow_event,nb_overflow_event);
  
  is_in_handler = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" void
_arcaneProfilingSigFunc(int signum)
{
  //cout << "**SIGPROF=" << global_is_active << "\n";
  if (signum!=SIGPROF)
    return;
  if (global_is_active){
    arcane_prof_handler();
    // It is preferable to position the timer once
    // the profiling function is called because if the timer is small,
    // it can trigger in the loop
    _setTimer(global_timer_period);
  }
}

extern "C" void
_arcaneProfilingSigactionFunc(int val, siginfo_t*,void*)
{
  _arcaneProfilingSigFunc(val);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
initialize()
{
  if (m_is_initialized)
    return;
  m_is_initialized = true;

  if (!global_infos)
    global_infos = new ProfInfos(traceMng());
  global_infos->setFunctionDepth(4);
  global_infos->setNbEventBeforeGettingStack(100);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
startProfiling()
{
  global_is_active = true;
  global_infos->startProfiling();

  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = _arcaneProfilingSigactionFunc;
  sigaction(SIGPROF, &sa, nullptr);

  _setTimer(global_timer_period);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
switchEvent()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
stopProfiling()
{
  if (!global_infos)
    return;
  global_is_active = false;
  _setTimer(global_timer_period);

  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = SIG_IGN;
  sigaction(SIGPROF, &sa, nullptr);

  //info() << "PROFILING: stop profiling nb_total=" << nb_total;
  global_infos->stopProfiling();
  //global_infos->printInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
printInfos(bool dump_file)
{
  if (global_infos)
    global_infos->printInfos(dump_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
dumpJSON(JSONWriter& writer)
{
  if (global_infos)
    global_infos->dumpJSON(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
getInfos(Int64Array&)
{
  throw NotImplementedException(A_FUNCINFO);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfPerformanceService::
reset()
{
  if (global_infos)
    global_infos->reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
