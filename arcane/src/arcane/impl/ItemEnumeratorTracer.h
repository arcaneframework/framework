// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumeratorTracer.h                                      (C) 2000-2022 */
/*                                                                           */
/* Trace les appels aux énumérateur sur les entités.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_ITEMENUMERATORTRACER_H
#define ARCANE_IMPL_ITEMENUMERATORTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un traceur d'énumérateur sur les entités.
 */
class ARCANE_IMPL_EXPORT ItemEnumeratorTracer
: public TraceAccessor
, public IItemEnumeratorTracer
{
 public:

  ItemEnumeratorTracer(ITraceMng* tm, IPerformanceCounterService* perf_counter);

 public:

  virtual ~ItemEnumeratorTracer();

 public:

  void enterEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti) override;
  void enterEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti) override;

 public:

  void dumpStats() override;
  IPerformanceCounterService* perfCounter() override { return m_perf_counter; }

 private:

  Int64 m_nb_call;
  Int64 m_nb_loop;
  IPerformanceCounterService* m_perf_counter;

 private:

  void _beginLoop(EnumeratorTraceInfo& eti);
  void _endLoop(EnumeratorTraceInfo& eti);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
