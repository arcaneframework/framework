// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnumeratorTracer.h                                          (C) 2000-2022 */
/*                                                                           */
/* Tracage des énumérateurs sur les composants.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_ENUMERATORTRACER_H
#define ARCANE_MATERIALS_ENUMERATORTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/materials/IEnumeratorTracer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT EnumeratorTracer
: public TraceAccessor
, public IEnumeratorTracer
{
 public:

  EnumeratorTracer(IPerformanceCounterService* perf_service, ITraceMng* tm);
  ~EnumeratorTracer();

 public:

  void enterEnumerator(const ComponentEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const ComponentEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const MatEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const MatEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const EnvEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const EnvEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const ComponentCellEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const ComponentCellEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const AllEnvCellEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const AllEnvCellEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const CellComponentCellEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const CellComponentCellEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const ComponentPartSimdCellEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const ComponentPartSimdCellEnumerator& e, EnumeratorTraceInfo& eti) override;

  void enterEnumerator(const ComponentPartCellEnumerator& e, EnumeratorTraceInfo& eti, const TraceInfo* ti) override;
  void exitEnumerator(const ComponentPartCellEnumerator& e, EnumeratorTraceInfo& eti) override;

 public:

  void dumpStats() override;

 private:

  Int64 m_nb_call = 0;
  Int64 m_nb_loop_component_cell = 0;
  Int64 m_nb_loop_cell_component_cell = 0;
  Int64 m_nb_loop_all_env_cell = 0;

  Int64 m_nb_call_component_cell = 0;
  Int64 m_nb_call_cell_component_cell = 0;
  Int64 m_nb_call_all_env_cell = 0;

  IPerformanceCounterService* m_perf_counter;

 private:

  void _beginLoop(EnumeratorTraceInfo& eti);
  void _endLoop(EnumeratorTraceInfo& eti);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
