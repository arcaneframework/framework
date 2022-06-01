// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnumeratorTracer.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Enumérateurs sur les mailles matériaux.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IPerformanceCounterService.h"
#include "arcane/materials/EnumeratorTracer.h"
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
EnumeratorTracer(IPerformanceCounterService* perf_service,ITraceMng* tm)
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
enterEnumerator(const ComponentEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ARCANE_UNUSED(e);
  if (ti)
    info() << "Enum size=" << e.m_size << " where=" << *ti;
  ++m_nb_call;
  m_perf_counter->getCounters(eti.counters(),false);
}
void EnumeratorTracer::
exitEnumerator(const ComponentEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  m_perf_counter->getCounters(eti.counters(),true);
  info() << "EndLoop: Component counters=" << eti.counters();
}

void EnumeratorTracer::
enterEnumerator(const MatEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(ti);
  ARCANE_UNUSED(eti);
  ++m_nb_call;
}

void EnumeratorTracer::
exitEnumerator(const MatEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
}

void EnumeratorTracer::
enterEnumerator(const EnvEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(ti);
  ARCANE_UNUSED(eti);
  ++m_nb_call;
}

void EnumeratorTracer::
exitEnumerator(const EnvEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
}

void EnumeratorTracer::
enterEnumerator(const ComponentCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ++m_nb_call_component_cell;
  m_nb_loop_component_cell += e.m_size;
  if (ti)
    info() << "ComponentCell size=" << e.m_size << " where=" << *ti;
  m_perf_counter->getCounters(eti.counters(),false);
}

void EnumeratorTracer::
exitEnumerator(const ComponentCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  m_perf_counter->getCounters(eti.counters(),true);
  info() << "EndLoop: ComponentCell counters=" << eti.counters();
}

void EnumeratorTracer::
enterEnumerator(const AllEnvCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ++m_nb_call_all_env_cell;
  m_nb_loop_all_env_cell += e.m_size;

  if (ti)
    info() << "ComponentCell size=" << e.m_size << " where=" << *ti;
  m_perf_counter->getCounters(eti.counters(),false);
}

void EnumeratorTracer::
exitEnumerator(const AllEnvCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  m_perf_counter->getCounters(eti.counters(),true);
  info() << "EndLoop: AllEnvCell counters=" << eti.counters();
}

void EnumeratorTracer::
enterEnumerator(const CellComponentCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ARCANE_UNUSED(ti);
  ARCANE_UNUSED(eti);
  ++m_nb_call_cell_component_cell;
  m_nb_loop_cell_component_cell += e.m_size;
}

void EnumeratorTracer::
exitEnumerator(const CellComponentCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
}

void EnumeratorTracer::
enterEnumerator(const ComponentPartSimdCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
  ARCANE_UNUSED(ti);
}

void EnumeratorTracer::
exitEnumerator(const ComponentPartSimdCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
}

void EnumeratorTracer::
enterEnumerator(const ComponentPartCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
  ARCANE_UNUSED(ti);
}

void EnumeratorTracer::
exitEnumerator(const ComponentPartCellEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  ARCANE_UNUSED(eti);
}

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
