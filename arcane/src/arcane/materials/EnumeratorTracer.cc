// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnumeratorTracer.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Enumérateurs sur les mailles matériaux.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/EnumeratorTracer.h"

#include "arcane/utils/IPerformanceCounterService.h"
#include "arcane/utils/Profiling.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ForLoopTraceInfo.h"

#include "arcane/materials/MatItemEnumerator.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnumeratorTracer::
EnumeratorTracer(ITraceMng* tm, Ref<IPerformanceCounterService> perf_service)
: TraceAccessor(tm)
, m_perf_counter(perf_service)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnumeratorTracer::
~EnumeratorTracer()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
_beginLoop(EnumeratorTraceInfo& eti)
{
  ++m_nb_call;
  m_perf_counter->getCounters(eti.counters(), false);
  eti.setBeginTime(platform::getRealTimeNS());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
_endLoop(EnumeratorTraceInfo& eti)
{
  m_perf_counter->getCounters(eti.counters(), true);
  const TraceInfo* ti = eti.traceInfo();
  ForLoopTraceInfo loop_trace_info;
  if (ti)
    loop_trace_info = ForLoopTraceInfo(*ti);
  ForLoopOneExecStat exec_stat;
  exec_stat.setBeginTime(eti.beginTime());
  exec_stat.setEndTime(platform::getRealTimeNS());
  ProfilingRegistry::_threadLocalForLoopInstance()->merge(exec_stat, loop_trace_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const ComponentEnumerator& e, EnumeratorTraceInfo& eti)
{
  _beginLoop(eti);
  const TraceInfo* ti = eti.traceInfo();
  if (ti && m_is_verbose)
    info() << "Enum size=" << e.m_size << " where=" << *ti;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const ComponentEnumerator&, EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
  if (m_is_verbose)
    info() << "EndLoop: Component counters=" << eti.counters();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const MatEnumerator&,EnumeratorTraceInfo& eti)
{
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const MatEnumerator&,EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const EnvEnumerator&,EnumeratorTraceInfo& eti)
{
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const EnvEnumerator&,EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const ComponentCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ++m_nb_call_component_cell;
  m_nb_loop_component_cell += e.m_size;
  const TraceInfo* ti = eti.traceInfo();
  if (ti && m_is_verbose)
    info() << "ComponentCell size=" << e.m_size << " where=" << *ti;
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const ComponentCellEnumerator&, EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
  if (m_is_verbose)
    info() << "EndLoop: ComponentCell counters=" << eti.counters();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const AllEnvCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ++m_nb_call_all_env_cell;
  m_nb_loop_all_env_cell += e.m_size;

  const TraceInfo* ti = eti.traceInfo();
  if (ti && m_is_verbose)
    info() << "ComponentCell size=" << e.m_size << " where=" << *ti;
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const AllEnvCellEnumerator&,EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
  if (m_is_verbose)
    info() << "EndLoop: AllEnvCell counters=" << eti.counters();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const CellComponentCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ++m_nb_call_cell_component_cell;
  m_nb_loop_cell_component_cell += e.m_size;
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const CellComponentCellEnumerator&,EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const ComponentPartSimdCellEnumerator&,EnumeratorTraceInfo& eti)
{
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const ComponentPartSimdCellEnumerator&,EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
enterEnumerator(const ComponentPartCellEnumerator&,EnumeratorTraceInfo& eti)
{
  _beginLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
exitEnumerator(const ComponentPartCellEnumerator&,EnumeratorTraceInfo& eti)
{
  _endLoop(eti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnumeratorTracer::
dumpStats()
{
  info() << "ENUMERATOR_TRACER_DUMP_STATS nb_call=" << m_nb_call;
  info() << " nb_call_all_env_cell=" << m_nb_call_all_env_cell
         << " nb_loop_all_env_cell=" << m_nb_loop_all_env_cell
         << " ratio=" << (Real)m_nb_loop_all_env_cell / (Real)(m_nb_call_all_env_cell+1);
  info() << " nb_call_component_cell=" << m_nb_call_component_cell
         << " nb_loop_component_cell=" << m_nb_loop_component_cell
         << " ratio=" << (Real)m_nb_loop_component_cell / (Real)(m_nb_call_component_cell+1);
  info() << " nb_call_cell_component_cell=" << m_nb_call_cell_component_cell
         << " nb_loop_cell_component_cell=" << m_nb_loop_cell_component_cell
         << " ratio=" << (Real)m_nb_loop_cell_component_cell / (Real)(m_nb_call_cell_component_cell+1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
