// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumeratorTracer.cc                                     (C) 2000-2022 */
/*                                                                           */
/* Trace les appels aux énumérateur sur les entités.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IPerformanceCounterService.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Profiling.h"
#include "arcane/utils/ForLoopTraceInfo.h"

#include "arcane/ItemEnumerator.h"
#include "arcane/SimdItem.h"

#include "arcane/impl/ItemEnumeratorTracer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT Ref<IItemEnumeratorTracer>
arcaneCreateItemEnumeratorTracer(ITraceMng* tm,IPerformanceCounterService* perf_counter)
{
  return Arccore::makeRef<IItemEnumeratorTracer>(new ItemEnumeratorTracer(tm,perf_counter));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemEnumeratorTracer::
ItemEnumeratorTracer(ITraceMng* tm,IPerformanceCounterService* perf_counter)
: TraceAccessor(tm)
, m_nb_call(0)
, m_nb_loop(0)
, m_perf_counter(perf_counter)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemEnumeratorTracer::
~ItemEnumeratorTracer()
{
  delete m_perf_counter;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
_beginLoop(EnumeratorTraceInfo& eti)
{
  ++m_nb_call;
  m_perf_counter->getCounters(eti.counters(), false);
  eti.setBeginTime(platform::getRealTimeNS());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
_endLoop(EnumeratorTraceInfo& eti)
{
  m_perf_counter->getCounters(eti.counters(), true);
  const TraceInfo* ti = eti.traceInfo();
  ForLoopTraceInfo loop_trace_info;
  if (ti)
    loop_trace_info = ForLoopTraceInfo(*ti);
  ForLoopOneExecStat exec_stat;
  exec_stat.setExecTime(platform::getRealTimeNS() - eti.beginTime());
  ProfilingRegistry::threadLocalInstance()->merge(exec_stat, loop_trace_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
enterEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti)
{
  m_nb_loop += e.count();
  Int64 begin_time = platform::getRealTimeNS();
  const TraceInfo* ti = eti.traceInfo();
  if (ti)
    info() << "Loop:" << (*ti) << " count=" << e.count() << " begin_time=" << begin_time;
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
exitEnumerator(const ItemEnumerator&, EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
  info() << "EndLoop: cycle=" << eti.counters()[0] << " fp=" << eti.counters()[1]
         << " L2DCM=" << eti.counters()[2];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
enterEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti)
{
  const TraceInfo* ti = eti.traceInfo();
  if (ti)
    info() << "SimdLoop:" << (*ti) << " count=" << e.count();
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
exitEnumerator(const SimdItemEnumeratorBase&, EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
  info() << "EndSimdLoop: cycle=" << eti.counters()[0] << " fp=" << eti.counters()[1]
         << " L2DCM=" << eti.counters()[2];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
dumpStats()
{
  info() << "ITEM_ENUMERATOR_TRACER Stats";
  info() << " nb_call=" << m_nb_call
         << " nb_loop=" << m_nb_loop
         << " ratio=" << (Real)m_nb_loop / (Real)(m_nb_call + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
