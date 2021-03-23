// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PerfmonPerformanceService.cc                                (C) 2000-2006 */
/*                                                                           */
/* Informations de performances utilisant Perfmon.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"

#include <map>
#include <set>

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdio.h>
#include <cxxabi.h>

#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>

// Il faut utiliser le chemin complet pour éviter des conflits avec la version
// installée par papi
#include <perfmon/perfmon.h>
// Dans la version que nous utilisons, il y a un bug dans le .h en c++
// (il manque les lignes suivantes)
//#ifdef __cplusplus
//}
//#endif
#include <perfmon/pfmlib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Session.
 */
class PerfmonPerformanceService
: public AbstractService
, public IProfilingService
{
 public:

  PerfmonPerformanceService(const ServiceBuildInfo& sbi);
  virtual ~PerfmonPerformanceService();

 public:

  virtual void startProfiling();
  virtual void switchEvent();
  virtual void stopProfiling();

 public:

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(PerfmonPerformanceService,
                                    IProfilingService,
                                    PerfmonProfilingService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PerfmonPerformanceService::
PerfmonPerformanceService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PerfmonPerformanceService::
~PerfmonPerformanceService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const int MAX_COUNTER = 3;
static const int MAX_STACK = 8;
static const int MAX_FUNC = 10000;
static const int MAX_FUNC_LEN = 200;

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS
typedef long long long_long;
#define SMPL_PERIOD	250000UL
static volatile unsigned long notification_received;
/*
 * This array must at least contain 2 events
 */
static char *event_list[]={
	"cpu_cycles",
	"IA64_INST_RETIRED",
};
#define N_EVENTS	sizeof(event_list)/sizeof(const char *)

static pfarg_reg_t pd[NUM_PMDS];
static int ctx_fd;

class PerfmonFuncInfo
{
 public:
  PerfmonFuncInfo() : m_do_stack(false)
    {
      m_func_name[0]='\0';
      for( Integer i=0; i<MAX_COUNTER; ++i )
        m_counters[i] = 0;
    }
 public:
  bool m_do_stack;
  long_long m_counters[MAX_COUNTER];
  char m_func_name[MAX_FUNC_LEN+1];
};

class PerfmonStackInfo
{
 public:
  PerfmonStackInfo()
    {
      for( Integer i=0; i<MAX_STACK; ++i )
        m_funcs_info[i] = 0;
    }
 public:
 bool operator<(const PerfmonStackInfo& pfi) const
    {
      return ::memcmp(m_funcs_info,pfi.m_funcs_info,MAX_STACK*sizeof(PerfmonFuncInfo*))<0;
    }
 public:
  PerfmonFuncInfo* m_funcs_info[MAX_STACK];
};


class PerfmonAddrInfo
{
 public:
 public:
  PerfmonAddrInfo() : m_func_info(0)
    {
      for( Integer i=0; i<MAX_COUNTER; ++i )
        m_counters[i] = 0;
    }
 public:
  long_long m_counters[MAX_COUNTER];
  PerfmonFuncInfo* m_func_info;
};

class PerfmonInfos
: public TraceAccessor
{
 public:
  PerfmonInfos(ITraceMng* tm)
  : TraceAccessor(tm), m_total_event(0), m_total_stack(0),
    m_is_running(false), m_current_func_info(0)
    {
      for( Integer i=0; i<MAX_COUNTER; ++i )
        m_counters[i] = 0;
    }
 public:
  typedef std::map<void*,PerfmonAddrInfo> AddrMap;
  typedef std::map<unw_word_t,PerfmonFuncInfo*> FuncMap;
  typedef std::map<PerfmonStackInfo,Int64> StackMap;
 public:
  void stopProfiling()
    {
      if (m_is_running){
        pfm_self_stop(ctx_fd);
        //int retval = PAPI_stop(m_event_set,0);
        //if (retval!=PAPI_OK){
        //fatal() << "** ERROR in papi_stop r=" << retval << '\n';
        //}
        m_is_running = false;
      }
    }
  void startProfiling()
    {
      if (m_is_running)
        return;
      pfm_self_start(ctx_fd);
      //int retval = PAPI_start(m_event_set);
      //if (retval!=PAPI_OK){
      //fatal() << "** ERROR in papi_start r=" << retval << '\n';
      //}
      m_is_running = true;
    }
 public:
  AddrMap m_addr_map;
  FuncMap m_func_map;
  StackMap m_stack_map;
  Int64 m_total_event;
  Int64 m_total_stack;
  long_long m_counters[MAX_COUNTER];
  bool m_is_running;
  int m_current_func_info;
  PerfmonFuncInfo m_func_info_buffer[MAX_FUNC];

 public:
};

static PerfmonInfos* global_infos = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static void
arcane_sigio_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	pfm_msg_t msg;
	int fd = ctx_fd;
	int r;
  
	//if (fd != ctx_fd) {
  //fprintf(stderr,"handler does not get valid file descriptor\n");
  //exit(-1);
  //}

	if (perfmonctl(fd, PFM_READ_PMDS, pd+1, 1) == -1) {
    fprintf(stderr,"PFM_READ_PMDS: %s", strerror(errno));
    exit(-1);
	}

	/*r = read(fd, &msg, sizeof(msg));
	if (r != sizeof(msg)) {
		fprintf(stderr,"cannot read overflow message: %s\n", strerror(errno));
    exit(-1);
    }*/

	/*
	 * XXX: risky to do printf() in signal handler!
	 */
	//printf("Notification %lu: %d %s\n",
  //       notification_received,
  // /      pd[1].reg_value,
  //       event_list[1]);

	/*
	 * At this point, the counter used for the sampling period has already
	 * be reset by the kernel because we are in non-blocking mode, self-monitoring.
	 */
  //String s = platform::getStackTraceService()->stackTrace();
  //char buf[128];
  //sprintf(buf,"%lx",sc->sc_ip);
  //cerr << "TRACE: ADDR=" << sc->sc_ip << " buf=" << buf << " trace=" << s << '\n';
  void* address = (void*)sc->sc_ip;

  static int nb_total = 0;
  static bool is_in_handler = false;
  if (is_in_handler)
    return;
  is_in_handler = true;
  ++nb_total;

#if 0
  bool is_counter0 = false;
  int overflow_event[MAX_COUNTER];
  int nb_overflow_event = MAX_COUNTER;
  PAPI_get_overflow_event_index(EventSet,overflow_vector,overflow_event,&nb_overflow_event);
  for( int i=0; i<nb_overflow_event; ++i ){
    if (overflow_event[i]==0)
      is_counter0 = true;
    if (overflow_event[i]<0 || overflow_event[i]>=MAX_COUNTER)
      cerr << "arcane_papi_handler: EVENT ERROR n=" << overflow_event[i] << '\n';
  }
#endif
  int overflow_event[1];
  overflow_event[0] = 0;
  int nb_overflow_event = 1;

  bool do_stack = false;
  bool do_add = false;
  bool is_counter0 = true;

  PerfmonInfos::AddrMap::iterator v = global_infos->m_addr_map.find(address);
  if (v!=global_infos->m_addr_map.end()){
    PerfmonAddrInfo& ai = v->second;
    ++global_infos->m_total_event;
    for( int i=0; i<nb_overflow_event; ++i )
      ++ai.m_counters[ overflow_event[i] ];
    if (ai.m_func_info){
      for( int i=0; i<nb_overflow_event; ++i )
        ++ai.m_func_info->m_counters[ overflow_event[i] ];
      if (global_infos->m_total_event>10000){
        if ((ai.m_func_info->m_counters[0]*30)>global_infos->m_total_event){
          ai.m_func_info->m_do_stack = true;
          if (is_counter0)
            do_stack = true;
        }
      }
    }
  }
  else
    do_add = true;
  
  //do_add = true;
  //do_stack = true;

  if (do_add || do_stack){
    PerfmonAddrInfo papi_address_info;
    PerfmonStackInfo papi_stack_info;
    for( int i=0; i<nb_overflow_event; ++i )
      ++papi_address_info.m_counters[ overflow_event[i] ];
#define FUNC_INDEX 1   
    unw_proc_info_t proc_info;
    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip, sp;
    unw_word_t offset;
    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    int index = 0;
    while (unw_step(&cursor) > 0 && index<(MAX_STACK+FUNC_INDEX)) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      unw_get_reg(&cursor, UNW_REG_SP, &sp);
      unw_get_proc_info(&cursor,&proc_info);
      unw_word_t proc_start = proc_info.start_ip;

      if (index>=FUNC_INDEX){
        PerfmonInfos::FuncMap::iterator func = global_infos->m_func_map.find(proc_start);
        PerfmonFuncInfo* papi_func_info = 0;
        if (func==global_infos->m_func_map.end()){
          if (global_infos->m_current_func_info>=MAX_FUNC){
            cerr << "arcane_papi_handler: MAX_FUNC reached !\n";
            break;
          }
          papi_func_info = &global_infos->m_func_info_buffer[global_infos->m_current_func_info];
          ++global_infos->m_current_func_info;
          papi_address_info.m_func_info = papi_func_info;
          unw_get_proc_name(&cursor,papi_func_info->m_func_name,MAX_FUNC_LEN,&offset);
          global_infos->m_func_map.insert(PerfmonInfos::FuncMap::value_type(proc_start,papi_func_info));
        }
        else{
          papi_func_info = func->second;
        }

        if (index<(MAX_STACK+FUNC_INDEX))
          papi_stack_info.m_funcs_info[index-FUNC_INDEX] = papi_func_info;
        if (index==FUNC_INDEX){
          //cerr << "** FUNC_INDEX=" << papi_func_info->m_func_name << "\n";
          papi_address_info.m_func_info = papi_func_info;
          if (!papi_func_info->m_do_stack || !is_counter0)
            break;
          else
            do_stack = true;
        }
      }
      ++index;
    }
    if (do_stack){
      ++global_infos->m_total_stack;
      PerfmonInfos::StackMap::iterator st = global_infos->m_stack_map.find(papi_stack_info);
      if (st!=global_infos->m_stack_map.end()){
        ++st->second;
      }
      else
        global_infos->m_stack_map.insert(PerfmonInfos::StackMap::value_type(papi_stack_info,1));
    }
    if (do_add)
      global_infos->m_addr_map.insert(PerfmonInfos::AddrMap::value_type(address,papi_address_info));
  }
  is_in_handler = false;

	/*
	 * And resume monitoring
	 */
	if (perfmonctl(fd, PFM_RESTART,NULL, 0) == -1) {
    //if (pfm_restart(fd)){
		cerr << "PFM_RESTART: " << strerror(errno) << '\n';
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PerfmonPerformanceService::
startProfiling()
{
  info() << " START PROFILING\n";

  if (!global_infos)
    global_infos = new PerfmonInfos(traceMng());

  int retval;
  caddr_t start,end;
  unsigned long length;
  unsigned int *profbuf;

	char **p;
	pfarg_context_t ctx[1];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_reg_t pc[NUM_PMCS];
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	struct sigaction act;
	unsigned int i;
	int ret;

  int sampling_period = 1000000;
  String s = platform::getEnvironmentVariable("ARCANE_PROFILING_SAMPLING_PERIOD");
  if (!s.null()){
    info() << " SAMPLING PERIOD FROM ENVIRONMENT '" << s << "'\n";
    Integer sp = 0;
    bool is_bad = builtInGetValue(sp,s);
    if (!is_bad && sp>100000){
      sampling_period = sp;
    }
  }
  info() << " SAMPLING_PERIOD '" << sampling_period << "'\n";

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal() << "pfm_initialize() failed. Can't initialize library\n";
	}

	/*
	 * Install the signal handler (SIGIO)
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)arcane_sigio_handler;
	sigaction (SIGPROF, &act, 0);

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	memset(pc, 0, sizeof(pc));
	memset(ctx, 0, sizeof(ctx));
	memset(&load_args, 0, sizeof(load_args));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	p = event_list;
	for (i=0; i < N_EVENTS ; i++, p++) {
		if (pfm_find_event(*p, &inp.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal() << "Cannot find " << *p << "event\n";
		}
	}

	/*
	 * set the default privilege mode for all counters:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm = PFM_PLM3;

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS) {
		fatal() << "Cannot configure events: " << pfm_strerror(ret);
	}

	/*
	 * when we know we are self-monitoring and we have only one context, then
	 * when we get an overflow we know where it is coming from. Therefore we can
	 * save the call to the kernel to extract the notification message. By default,
	 * a message is generated. The queue of messages has a limited size, therefore
	 * it is important to clear the queue by reading the message on overflow. Failure
	 * to do so may result in a queue full and you will lose notification messages.
	 *
	 * With the PFM_FL_OVFL_NO_MSG, no message will be queue, but you will still get
	 * the signal. Similarly, the PFM_MSG_END will be generated.
	 */
	ctx->ctx_flags = PFM_FL_OVFL_NO_MSG;
	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal() << "Your kernel does not have performance monitoring support!";
		}
		fatal() << "Can't create PFM context " << strerror(errno);
	}
	ctx_fd = ctx->ctx_fd;

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of registers to
	 * setup. Note that this field can be >= pfp_event_count.
	 */

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	for (i=0; i < inp.pfp_event_count; i++) {
		pd[i].reg_num   = pc[i].reg_num;
	}
	/*
	 * We want to get notified when the counter used for our first
	 * event overflows
	 */
	pc[0].reg_flags 	|= PFM_REGFL_OVFL_NOTIFY;
	pc[0].reg_reset_pmds[0] |= 1UL << outp.pfp_pmcs[1].reg_num;

	/*
	 * we arm the first counter, such that it will overflow
	 * after SMPL_PERIOD events have been observed
	 */
	pd[0].reg_value       = (~0UL) - SMPL_PERIOD + 1;
	pd[0].reg_long_reset  = (~0UL) - SMPL_PERIOD + 1;
	pd[0].reg_short_reset = (~0UL) - SMPL_PERIOD + 1;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more than counting monitors.
	 */
	if (perfmonctl(ctx_fd, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count) == -1) {
		fatal() << "perfmonctl error PFM_WRITE_PMCS errno " << errno;
	}

	if (perfmonctl(ctx_fd, PFM_WRITE_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal() << "perfmonctl error PFM_WRITE_PMDS errno " << errno;
	}

	/*
	 * we want to monitor ourself
	 */
	load_args.load_pid = getpid();

	if (perfmonctl(ctx_fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
		fatal() << "perfmonctl error PFM_WRITE_PMDS errno " << errno;
	}

	/*
	 * setup asynchronous notification on the file descriptor
	 */
	ret = fcntl(ctx_fd, F_SETFL, fcntl(ctx_fd, F_GETFL, 0) | O_ASYNC);
	if (ret == -1) {
		fatal() << "cannot set ASYNC: " <<  strerror(errno);
	}

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(ctx_fd, F_SETOWN, getpid());
	if (ret == -1) {
		fatal() << "cannot setown: " << strerror(errno);
	}

	/*
	 * when you explicitely declare that you want a particular signal,
	 * even with you use the default signal, the kernel will send more
	 * information concerning the event to the signal handler.
	 *
	 * In particular, it will send the file descriptor from which the
	 * event is originating which can be quite useful when monitoring
	 * multiple tasks from a single thread.
	 */
	ret = fcntl(ctx_fd, F_SETSIG, SIGPROF);
	if (ret == -1)
		fatal() << "cannot setsig: " << strerror(errno);

  global_infos->startProfiling();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PerfmonPerformanceService::
switchEvent()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class PerfmonFuncComparer
{
 public:
  typedef PerfmonFuncInfo* PerfmonFuncInfoPtr;
public:
  bool operator()(const PerfmonFuncInfoPtr& lhs,const PerfmonFuncInfoPtr& rhs)
    {
      return (lhs->m_counters[0]>rhs->m_counters[0]);
    }
};

void PerfmonPerformanceService::
stopProfiling()
{
  if (!global_infos)
    return;
  info() << " STOP PROFILING\n";
  global_infos->stopProfiling();

  Int64 total_event = 1;
  {
    info() << " NB ADDRESS MAP = " << global_infos->m_addr_map.size();
    info() << " NB FUNC MAP = " << global_infos->m_func_map.size();
    info() << " NB STACK MAP = " << global_infos->m_stack_map.size();
    info() << " TOTAL STACK = " << global_infos->m_total_stack;
    {
      PerfmonInfos::AddrMap::const_iterator begin = global_infos->m_addr_map.begin();
      for( ; begin!=global_infos->m_addr_map.end(); ++begin ){
        PerfmonFuncInfo* pf = begin->second.m_func_info;
        Int64 nb_event = begin->second.m_counters[0];
        total_event += nb_event;
      }
    }

    {
      String file_name = "profiling.addr";
      file_name += platform::getProcessId();
      file_name += ".xml";
      ofstream ofile(file_name.localstr());
      ofile << "<?xml version='1.0'?>\n";
      ofile << "<addresses>\n";
      PerfmonInfos::AddrMap::const_iterator begin = global_infos->m_addr_map.begin();
      for( ; begin!=global_infos->m_addr_map.end(); ++begin ){
        void* addr = begin->first;
        const PerfmonAddrInfo& pa = begin->second;
        ofile << "<addr addr='" << addr << "'"
              << " count='" << pa.m_counters[0] << "'"
              << " func='" << pa.m_func_info->m_func_name << "'"
              << ">\n";
      }
      ofile << "/<addresses>\n";
    }
  }
  {
    std::set<PerfmonFuncInfo*,PerfmonFuncComparer> sorted_func;
    {
      PerfmonInfos::FuncMap::const_iterator begin = global_infos->m_func_map.begin();
      for( ; begin!=global_infos->m_func_map.end(); ++begin ){
        if (begin->second)
          sorted_func.insert(begin->second);
      }
    }

    std::set<PerfmonFuncInfo*,PerfmonFuncComparer>::const_iterator begin = sorted_func.begin();
    char demangled_func_name[512];
    info() << " TOTAL EVENT = " << total_event;
    info() << " event     %   function";
    Integer index = 0;
    for( ; begin!=sorted_func.end(); ++begin ){
      PerfmonFuncInfo* pfi = *begin;
      const char* func_name = pfi->m_func_name;
      size_t len = 512;
      int dstatus = 0;
      const char* buf = abi::__cxa_demangle(func_name,demangled_func_name,&len,&dstatus);
      if (!buf)
        buf = func_name;
      Int64 nb_event = pfi->m_counters[0];
      Int64 total_percent = (nb_event * 1000) / total_event;
      Int64 percent = (total_percent/10);
      info() << "  " << Trace::Width(10) << nb_event
             << "  " << Trace::Width(3) << percent << "." << (total_percent % 10)
             << " " << Trace::Width(12) << pfi->m_counters[0]
             << " " << Trace::Width(12) << pfi->m_counters[1]
             << " " << Trace::Width(12) << pfi->m_counters[2]
             << " " << pfi->m_do_stack
             << "  " << buf;
      if (total_percent<5 && index>20)
        break;
      ++index;
    }
  }
  {
    std::set<PerfmonFuncInfo*,PerfmonFuncComparer> sorted_func;
    {
      PerfmonInfos::FuncMap::const_iterator begin = global_infos->m_func_map.begin();
      for( ; begin!=global_infos->m_func_map.end(); ++begin ){
        if (begin->second)
          sorted_func.insert(begin->second);
      }
    }

    PerfmonInfos::StackMap::const_iterator begin = global_infos->m_stack_map.begin();
    char demangled_func_name[512];
    Integer index = 0;
    String file_name = "profiling.callstack";
    file_name += platform::getProcessId();
    ofstream ofile(file_name.localstr());
    for( ; begin!=global_infos->m_stack_map.end(); ++begin ){
      const PerfmonStackInfo& psi = begin->first;
      Integer nb_stack = begin->second;
      if (nb_stack<2)
        continue;
      ofile << " Stack nb=" << begin->second << '\n';
      for( Integer z=0; z<MAX_STACK; ++z ){
        PerfmonFuncInfo* pfi = psi.m_funcs_info[z];
        if (pfi){
          const char* func_name = pfi->m_func_name;
          size_t len = 512;
          int dstatus = 0;
          const char* buf = abi::__cxa_demangle(func_name,demangled_func_name,&len,&dstatus);
          if (!buf)
            buf = func_name;
          Int64 nb_event = pfi->m_counters[0];
          Int64 total_percent = (nb_event * 1000) / total_event;
          Int64 percent = (total_percent/10);
          ofile << "  " << buf << '\n';
        }
        else
          info() << " NO FUNC";
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
