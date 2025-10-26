// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExecutionStatsDumper.h                                      (C) 2000-2025 */
/*                                                                           */
/* Ecriture des statistiques d'exécution.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_EXECUTIONSTATSDUMPER_H
#define ARCANE_IMPL_EXECUTIONSTATSDUMPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace impl
{
class AcceleratorStatInfoList;
}

class ISimpleTableOutput;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture des statistiques d'exécution.
 *
 * Les statistiques sont sorties à la fois dans le listing et dans les
 * logs.
 */
class ExecutionStatsDumper
: public TraceAccessor
{
 public:

  explicit ExecutionStatsDumper(ITraceMng* trace)
  : TraceAccessor(trace)
  {}

 public:

  void dumpStats(ISubDomain* sd, ITimeStats* time_stats);

 private:

  void _dumpProfiling(std::ostream& o);
  void _dumpOneLoopListStat(std::ostream& o, const Impl::ForLoopStatInfoList& stat_list);
  void _printGlobalLoopInfos(std::ostream& o, const Impl::ForLoopCumulativeStat& cumulative_stat);
  void _dumpProfilingJSON(const String& filename);
  void _dumpProfilingJSON(JSONWriter& json_writer);
  void _dumpProfilingTable(ISimpleTableOutput* table);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
