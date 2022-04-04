// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaProf.cc                                                (C) 2000-2013 *
 *****************************************************************************/
#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/IServiceLoader.h"
#include "arcane/IServiceMng.h"
#include "arcane/ServiceRegisterer.h"
#include "arcane/IServiceFactory.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/AbstractService.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/std/ProfilingInfo.h"
#if 0
#include <set>
#ifdef ARCANE_HAS_PACKAGE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdio.h>
#endif
#ifdef __GNUG__
#include <cxxabi.h>
#endif
#if defined(ARCANE_OS_LINUX)
#define ARCANE_CHECK_MEMORY_USE_MALLOC_HOOK
#include <execinfo.h>
#include <dlfcn.h>
#endif
#endif

#include <arcane/hyoda/Hyoda.h>
#include <arcane/hyoda/HyodaArc.h>
#include <arcane/hyoda/HyodaTcp.h>
#include <arcane/hyoda/HyodaPapi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HyodaPapi::
HyodaPapi(Hyoda *hd,IApplication *app,ITraceMng *tm):
  TraceAccessor(tm),
  m_hyoda(hd),
  m_app(app),
  m_sub_domain(NULL),
  m_papi(NULL),
  m_tcp(NULL){
}

HyodaPapi::~HyodaPapi()
{
  if (!m_papi)
    return;
  m_papi->stopProfiling();
  debug()<<"\33[7m[HyodaPapi::~HyodaPapi]\33[m";
  delete m_papi;
  //m_papi->printInfos(false);
}


void HyodaPapi::initialize(ISubDomain *sd, HyodaTcp *tcp)
{
  if (m_papi){
    ServiceBuilder<IProfilingService> sb(m_app);
    m_papi = sb.createInstance("PapiProfilingService", SB_AllowNull);
  }
  if (!m_papi) return;
  m_sub_domain=sd;
  m_tcp=tcp;
  debug()<<"\33[7m[HyodaPapi::HyodaPapi] PAPI INIitializes\33[m";
  m_papi->initialize();
}

void HyodaPapi::start(void){
  if (!m_papi) return;
  debug()<<"\33[7m[HyodaPapi::HyodaPapi] PAPI STARTS profiling\33[m";
  m_papi->startProfiling();
}

void HyodaPapi::stop(void){
  if (!m_papi) return;
  debug()<<"\33[7m[HyodaPapi::HyodaPapi] PAPI STOPS profiling\33[m";
  m_papi->stopProfiling();
}


// ****************************************************************************
// * PAPI_TOT_CYC 0x8000003b  Yes   No   Total cycles
// * PAPI_RES_STL 0x80000039  Yes   No   Cycles stalled on any resource
// * PAPI_FP_INS  0x80000034  Yes   No   Floating point instructions
// ****************************************************************************
void HyodaPapi::dump(void){
  if (!m_papi) return;
  pkt.clear();
  pkt.add((Int64)0xb80dd1a3ul); // Header
  pkt.add((Int64)0);            // Size
  //m_papi->printInfos();
  m_papi->getInfos(pkt);
  debug()<<"\33[7m[HyodaPapi::dump] pkt size is "<<pkt.size()<<"\33[m";
  pkt[1]=pkt.size()-1;
  // Seul, le rang zéro s'occupe du packet
  if (m_sub_domain->parallelMng()->commRank()!=0) return;
  m_tcp->send((char*)pkt.unguardedBasePointer(), pkt.size()<<3);
  m_tcp->waitForAcknowledgment();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
