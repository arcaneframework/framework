// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumeratorTracer.cc                                     (C) 2000-2016 */
/*                                                                           */
/* Trace les appels aux énumérateur sur les entités.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IPerformanceCounterService.h"

#include "arcane/ItemEnumerator.h"
#include "arcane/SimdItem.h"

#include "arcane/impl/ItemEnumeratorTracer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IItemEnumeratorTracer*
arcaneCreateItemEnumeratorTracer(ITraceMng* tm,IPerformanceCounterService* perf_counter)
{
  return new ItemEnumeratorTracer(tm,perf_counter);
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
enterEnumerator(const ItemEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  ++m_nb_call;
  m_nb_loop += e.count();
  if (ti)
    info() << "Loop:" << (*ti) << " count=" << e.count();
  m_perf_counter->getCounters(eti.counters(),false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
exitEnumerator(const ItemEnumerator& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  m_perf_counter->getCounters(eti.counters(),true);
  info() << "EndLoop: cycle=" << eti.counters()[0] << " fp=" << eti.counters()[1]
         << " L2DCM=" << eti.counters()[2];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
enterEnumerator(const SimdItemEnumeratorBase& e,EnumeratorTraceInfo& eti,const TraceInfo* ti)
{
  if (ti)
    info() << "SimdLoop:" << (*ti) << " count=" << e.count();
  m_perf_counter->getCounters(eti.counters(),false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemEnumeratorTracer::
exitEnumerator(const SimdItemEnumeratorBase& e,EnumeratorTraceInfo& eti)
{
  ARCANE_UNUSED(e);
  m_perf_counter->getCounters(eti.counters(),true);
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
         << " ratio=" << (Real)m_nb_loop / (Real)(m_nb_call+1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
