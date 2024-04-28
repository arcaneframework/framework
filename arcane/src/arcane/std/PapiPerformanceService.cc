// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PapiPerformanceService.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Informations de performances utilisant PAPI.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/PapiPerformanceService.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/CriticalSection.h"
#include "arcane/utils/IPerformanceCounterService.h"

#include "arcane/FactoryService.h"
#include "arcane/IParallelSuperMng.h"

#include "arcane/std/ProfilingInfo.h"
#include "arcane/impl/TimerMng.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
/*
 * Vérifie si PAPI_library_init() a été appelé et si ce n'est pas le cas,
 * appelle cette méthode.
 *
 * A noter que PAPI_library_init() ne peut être appelé qu'une seule fois.
 */
void
_checkInitPAPI()
{
  if (PAPI_is_initialized()==PAPI_NOT_INITED){
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval!=PAPI_VER_CURRENT && retval>0)
      ARCANE_FATAL("PAPI version mismatch r={0} current={1}",retval,PAPI_VER_CURRENT);
    if (retval<0)
      ARCANE_FATAL("Error in PAPI_library_init r={0} msg={1}",retval,PAPI_strerror(retval));
  }
}

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(PapiPerformanceService,
                        ServiceProperty("PapiProfilingService",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IProfilingService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO: Avec les threads, utiliser une instance par sous-domaine
 * et utiliser PAPI_register_thread et PAPI_unregister_thread.
 * (voir exemple papi overflow_pthreads.c).
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PapiPerformanceService::
PapiPerformanceService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_period(500000)
, m_event_set(PAPI_NULL)
, m_application(sbi.application())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PapiPerformanceService::
~PapiPerformanceService()
{
  delete m_timer_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
ProfInfos* global_infos = nullptr;
int global_nb_total_call = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
arcane_papi_handler(int EventSet, void *address, long_long overflow_vector, void *context)
{
  ARCANE_UNUSED(context);
  static bool is_in_handler = false;
  // Sous Linux avec gcc, les exceptions utilisent la libunwind contenue
  // dans gcc et cela peut provoquer des deadlocks avec notre utilisation
  // si cet handler est appelé lors du dépilement d'une exception.
  // Pour éviter ce problème, on ne fait rien tant qu'une exception est
  // active.
  if (Exception::hasPendingException()){
    cout << "** WARNING: PapiHandler in pending exception\n";
    return;
  }
  if (is_in_handler)
    return;

  is_in_handler = true;
  ++global_nb_total_call;

  int overflow_event[MAX_COUNTER];
  int nb_overflow_event = MAX_COUNTER;
  PAPI_get_overflow_event_index(EventSet,overflow_vector,overflow_event,&nb_overflow_event);
  global_infos->addEvent(address,overflow_event,nb_overflow_event);
  is_in_handler = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PapiPerformanceService::
_addEvent(int event_code,int event_index)
{
  char event_name[PAPI_MAX_STR_LEN];
  //int event_code = PAPI_TOT_CYC;
  int retval = PAPI_add_event(m_event_set,event_code);
  PAPI_event_code_to_name(event_code,event_name);
  info() << "Adding Papi event name=" << event_name;
  if (retval!=PAPI_OK){
    pwarning() << "** ERROR in add_event (index=" << event_index << ") r=" << retval
               << " msg=" << PAPI_strerror(retval);
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
initialize()
{
  CriticalSection cs(m_application->parallelSuperMng()->threadMng());
  if (m_is_initialized)
    return;
  if (m_is_running)
    return;

  m_is_initialized = true;

  int major = PAPI_VERSION_MAJOR(PAPI_VERSION);
  int minor = PAPI_VERSION_MINOR(PAPI_VERSION);
  int sub_minor = PAPI_VERSION_REVISION(PAPI_VERSION);
  info() << "PROFILING: start profiling using 'PAPI' version="
         << major << "." << minor << "." << sub_minor;

  if (!global_infos)
    global_infos = new ProfInfos(traceMng());

  global_infos->setNbEventBeforeGettingStack(2000);
  global_infos->setFunctionDepth(5);

  int retval = 0;
  caddr_t start,end;
  const PAPI_exe_info_t *prginfo;

  _checkInitPAPI();

  retval = PAPI_thread_init((unsigned long (*)(void)) (pthread_self));
  if (retval!=PAPI_OK)
    ARCANE_FATAL("Error in PAPI_thread_init r={0} msg={1}",retval,PAPI_strerror(retval));

  prginfo = PAPI_get_executable_info();
  
  start = reinterpret_cast<caddr_t>(prginfo->address_info.text_start);
  end = reinterpret_cast<caddr_t>(prginfo->address_info.text_end);

  info() << "** PROGRAM INFOS: start=" << (long)start << " end=" << (long)end << " length=" << (end-start);

  if ((retval=PAPI_create_eventset(&m_event_set))!=PAPI_OK)
    ARCANE_FATAL("ERROR in PAPI_create_eventset r={0}",retval);

  const int NB_EVENT = 3;
  int papi_events[NB_EVENT];

  // L'évènement 0 doit toujours être le PAPI_TOT_CYC car on s'en sert
  // pour les statistiques
  papi_events[0] = PAPI_TOT_CYC;
  // TODO: regarder si ses évènements sont supportés par le proc
  papi_events[1] = PAPI_RES_STL;
  papi_events[2] = PAPI_DP_OPS;

  String papi_user_events = platform::getEnvironmentVariable("ARCANE_PAPI_EVENTS");
  int nb_event = NB_EVENT;
  if (!papi_user_events.null()){
    StringUniqueArray strs;
    papi_user_events.split(strs,':');
    int nb_str = strs.size();
    nb_str = math::min(NB_EVENT-1,nb_str);
    for( Integer i=0; i<nb_str; ++i ){
      const String& ename = strs[i];
      info() << "USER_EVENT name=" << ename;
      int new_event;
      int retval = PAPI_event_name_to_code((char*)ename.localstr(), &new_event);
      if (retval!=PAPI_OK){
        pwarning() << "Can not set event from name=" << ename << " r=" << retval;
      }
      else
        papi_events[i+1] = new_event;
    }
    nb_event = nb_str+1;
  }
  bool is_valid_event[NB_EVENT];

  for( Integer i=0; i<nb_event; ++i )
    is_valid_event[i] = _addEvent(papi_events[i],i);


  //int period = 500000;
  String period_str = platform::getEnvironmentVariable("ARCANE_PROFILING_PERIOD");
  if (!period_str.null()){
    bool is_bad = builtInGetValue(m_period,period_str);
    if (is_bad){
      pwarning() << "Can not convert '" << period_str << "' to int";
    }
  }
  String only_str = platform::getEnvironmentVariable("ARCANE_PROFILING_ONLYFLOPS");
  if (!only_str.null())
    m_only_flops = true;

  if (m_only_flops){
    _printFlops();
  }
  else{
    // Il ne faut pas que la période soit trop petite sinon on passe tout
    // le temps dans le traitement de 'arcane_papi_handler'.
    if (m_period<100000)
      m_period = 100000;
    for( Integer i=0; i<nb_event; ++i ){
      if (is_valid_event[i]){
        retval = PAPI_overflow(m_event_set, papi_events[i], m_period, 0, arcane_papi_handler);
        if (retval!=PAPI_OK){
          // L'évènement PAPI_TOT_CYC est indispensable
          if (i==0){
            fatal() << "** ERROR in papi_overflow i=" << i << " r=" << retval
                    << " msg=" << PAPI_strerror(retval);
          }
          else
            pwarning() << "** ERROR in papi_overflow i=" << i << " r=" << retval
                       << " msg=" << PAPI_strerror(retval);
          is_valid_event[i] = false;
        }
      }
    }
    info() << "Période de sampling: (en évènements) " << m_period;
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
startProfiling()
{
  if (m_only_flops)
    return;
  ARCANE_CHECK_POINTER(global_infos);
  global_infos->startProfiling();
  if (!m_is_running){
    int retval = PAPI_start(m_event_set);
    if (retval!=PAPI_OK){
      ARCANE_FATAL("** ERROR in papi_start r={0}",retval);
    }
    m_is_running = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
switchEvent()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
_printFlops()
{
  float real_time = 0.0f;
  float proc_time = 0.0f;
  long long flpins = 0.0;
  float mflops = 0.0f;
  // A partir de PAPI 6.0 il n'y a plus PAPI_flops mais à la place
  // c'est 'PAPI_flops_rate' mais il y a un argument supplémentaire
  // à mettre pour spécifier le type de flop à calculer (simple précision,
  // double précision, ...)
#if PAPI_VERSION >= PAPI_VERSION_NUMBER(6,0,0,0)
  int retval = PAPI_flops_rate(PAPI_DP_OPS, &real_time, &proc_time, &flpins, &mflops);
#else
  int retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
#endif
  if (retval!=PAPI_OK)
    error() << "** ERROR in PAPI_flops r=" << retval;
  else{
    info() << "PAPI_Flops: real_time=" << real_time
           << " proc_time=" << proc_time
           << " flpins=" << flpins
           << " mflips=" << mflops;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
stopProfiling()
{
  if (!global_infos)
    return;

  if (m_only_flops){
    _printFlops();
    return;
  }
  CriticalSection cs(m_application->parallelSuperMng()->threadMng());

  //info() << "PROFILING: stop profiling nb_call=" << global_nb_total_call;
  if (m_is_running){
    int retval = PAPI_stop(m_event_set,0);
    if (retval!=PAPI_OK){
      ARCANE_FATAL("** ERROR in papi_stop r={0}",retval);
    }
    m_is_running = false;
  }
  global_infos->stopProfiling();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
printInfos(bool dump_file)
{
  if (global_infos)
    global_infos->printInfos(dump_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
dumpJSON(JSONWriter& writer)
{
  if (global_infos)
    global_infos->dumpJSON(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
getInfos(Int64Array &array)
{
  if (global_infos)
    global_infos->getInfos(array);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiPerformanceService::
reset()
{
  if (global_infos)
    global_infos->reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PapiTimerMng
: public TimerMng
{
 public:
  explicit PapiTimerMng(ITraceMng* tm)
  : TimerMng(tm), m_nb_event(0), m_event_set(PAPI_NULL), m_is_started(false),
    m_is_init(false), m_elapsed_us(0), m_elapsed_cycle()
  {}
  ~PapiTimerMng()
  {
    if (m_is_started)
      PAPI_stop(m_event_set,m_values.data());
  }
 public:
  void init();
  void _addEvent(int event);
  void start();
  Real stop(const char* msg);

  //! Retourne le temps réel
  Real _getRealTime() override
  {
    return stop("test");
  }

  //! Positionne un timer réel
  void _setRealTime() override
  {
    if (!m_is_init){
      m_is_init = true;
      init();
      start();
    }
  }

 private:
  int m_nb_event;
  int m_event_set;
  bool m_is_started;
  bool m_is_init;
  UniqueArray<long_long> m_values;
  UniqueArray<long_long> m_start_values;
  long_long m_elapsed_us;
  long_long m_elapsed_cycle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiTimerMng::
init()
{
  _checkInitPAPI();
    
  int retval = PAPI_thread_init((unsigned long (*)(void)) (pthread_self));
  if (retval != PAPI_OK)
    ARCANE_FATAL("PAPI_thread_init r={0}",retval);
      
  retval = PAPI_create_eventset(&m_event_set);
  if (retval!=PAPI_OK)
    ARCANE_FATAL("PAPI_create_eventset r={0}",retval);

  _addEvent(PAPI_TOT_CYC);
  _addEvent(PAPI_DP_OPS);
  m_values.resize(m_nb_event);
  m_values.fill(0);
  m_start_values.resize(m_nb_event);
  m_start_values.fill(0);

  retval = PAPI_start(m_event_set);
  if (retval != PAPI_OK)
    ARCANE_FATAL("PAPI_start r={0}",retval);
  m_is_started = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiTimerMng::
_addEvent(int event)
{
  int retval = PAPI_add_event(m_event_set,event);
  if (retval!=PAPI_OK){
    cerr << "** CAN NOT FIND EVENT " << event << '\n';
    return;
  }
  ++m_nb_event;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PapiTimerMng::
start()
{
  int retval = PAPI_read(m_event_set,m_start_values.data());
  if (retval!=PAPI_OK){
    cerr << "** CAN NOT START EVENT\n";
  }
  m_elapsed_us = PAPI_get_real_usec();
  m_elapsed_cycle = PAPI_get_real_cyc();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real PapiTimerMng::
stop(const char* msg)
{
  int retval = PAPI_read(m_event_set,m_values.data());
  if (retval!=PAPI_OK){
    cerr << "** CAN NOT STOP EVENT\n";
  }
  long_long elapsed_us = PAPI_get_real_usec() - m_elapsed_us;
  //long_long elapsed_cycle = PAPI_get_real_cyc() - m_elapsed_cycle;

  double sec_time = ((double)elapsed_us) / 1.0e6;

  std::cout << " -- -- Time: ";
  std::cout.width(60);
  std::cout << msg << " = ";
  std::cout.width(10);
  std::cout << sec_time << " ";

  //cout << "** TIME: " << (elapsed_us) << " CYCLE=" << elapsed_cycle << '\n';

  long_long nb_cycle = m_values[0]-m_start_values[0];
  long_long nb_flop = m_values[1]-m_start_values[1];
      
  std::cout.width(15);
  std::cout << nb_cycle << " ";
  std::cout.width(12);
  std::cout << nb_flop;

  std::cout << " (";
  std::cout.width(5);
  std::cout << nb_flop/elapsed_us;
  std::cout << ")";

  std::cout << '\n';

  //for( Integer i=0, is=m_values.size() ; i<is; ++i ){
  //  long_long diff = (m_values[i]-m_start_values[i]);
  //  cout << "** EVENT: " << diff << " FLOP=" << (diff/elapsed_us) << '\n';
  //}
  return sec_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimerMng* PapiPerformanceService::
timerMng()
{
  if (!m_timer_mng)
    m_timer_mng = new PapiTimerMng(traceMng());
  return m_timer_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PapiPerformanceCounterService
: public TraceAccessor
, public IPerformanceCounterService
{
 public:
  PapiPerformanceCounterService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng()), m_nb_event(0),
    m_event_set(PAPI_NULL), m_is_started(false)
  {
  }
  ~PapiPerformanceCounterService()
  {
    if (m_is_started)
      (void)PAPI_stop(m_event_set,nullptr);
  }
 public:

  void build()
  {
  }

  void initialize() override
  {
    int retval = 0;

    _checkInitPAPI();

    retval = PAPI_thread_init((unsigned long (*)(void)) (pthread_self));
    if (retval != PAPI_OK)
      ARCANE_FATAL("Error in 'PAPI_thread_init' r={0}",retval);

    //int event_mask = MASK_FP_OPS | MASK_L2_TCM | MASK_TOT_CYC;
    //m_event_set = _makeEventSet(&num_events,&event_mask);
    retval = PAPI_create_eventset(&m_event_set);
    if (retval!=PAPI_OK)
      ARCANE_FATAL("Error in 'PAPI_createeventset' r={0}",retval);

    _addEvent(PAPI_TOT_CYC);
    _addEvent(PAPI_RES_STL);
    _addEvent(PAPI_L2_TCM);
  }
  void _addEvent(int event)
  {
    int retval = PAPI_add_event(m_event_set,event);
    if (retval!=PAPI_OK){
      error() << "** CAN NOT FIND EVENT " << event << '\n';
      return;
    }
    ++m_nb_event;
  }
  void start() final
  {
    if (m_is_started)
      ARCANE_FATAL("start() has alredy been called");
    int retval = PAPI_start(m_event_set);
    if (retval != PAPI_OK)
      ARCANE_FATAL("Error in 'PAPI_start' r={0}",retval);
    m_is_started = true;
  }
  void stop() final
  {
    if (!m_is_started)
      ARCANE_FATAL("start() has not been called");
    int retval = PAPI_stop(m_event_set,nullptr);
    if (retval != PAPI_OK)
      ARCANE_FATAL("Error in 'PAPI_stop' r={0}",retval);
    m_is_started = false;
  }
  bool isStarted() const final
  {
    return m_is_started;
  }

  Integer getCounters(Int64ArrayView counters,bool do_substract) final
  {
    long_long values[MIN_COUNTER_SIZE];
    int retval = PAPI_read(m_event_set,values);
    if (retval!=PAPI_OK){
      error() << "Error in 'PAPI_read' during getCounters() r=" << retval;
    }
    Integer n = m_nb_event;
    if (do_substract){
      for( int i=0; i<n; ++i )
        counters[i] = (Int64)values[i] - counters[i];
    }
    else
      for( int i=0; i<n; ++i )
        counters[i] = values[i];
    return n;
  }

  Int64 getCycles() final
  {
    std::array<Int64,MIN_COUNTER_SIZE> values;
    Int64ArrayView view(values);
    getCounters(view,false);
    return view[0];
  }

 private:

  int m_nb_event = 0;
  int m_event_set = 0;
  bool m_is_started = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(PapiPerformanceCounterService,
                        ServiceProperty("PapiPerformanceCounterService",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IPerformanceCounterService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
