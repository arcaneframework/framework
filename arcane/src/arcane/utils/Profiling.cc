// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiling.cc                                                (C) 2000-2022 */
/*                                                                           */
/* Classes pour gérer le profilage.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Profiling.h"

#include "arcane/utils/ForLoopTraceInfo.h"
#include "arcane/utils/PlatformUtils.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  impl::LoopStatInfo global_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ScopedStatLoop::
ScopedStatLoop(LoopStatInfo* s)
: m_stat_info(s)
{
  if (m_stat_info) {
    m_stat_info->incrementNbLoopParallelFor();
    m_begin_time = platform::getRealTime();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ScopedStatLoop::
~ScopedStatLoop()
{
  if (m_stat_info) {
    double v = platform::getRealTime() - m_begin_time;
    Int64 v_as_int64 = static_cast<Int64>(v * 1.0e9);
    m_stat_info->m_total_time += v_as_int64;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllStatInfoList
{
 public:

  StatInfoList* createStatInfoList()
  {
    std::lock_guard<std::mutex> lk(m_mutex);
    std::unique_ptr<StatInfoList> x(new StatInfoList);
    StatInfoList* ptr = x.get();
    m_stat_info_list_vector.push_back(std::move(x));
    return ptr;
  }

  void print(std::ostream& o)
  {
    for (const auto& x : m_stat_info_list_vector)
      x->print(o);
  }

 public:

  std::mutex m_mutex;
  std::vector<std::unique_ptr<StatInfoList>> m_stat_info_list_vector;
};
AllStatInfoList global_all_stat_info_list;

// Permet de gérer une instance de StatInfoList par thread pour éviter les verroux
class ThreadLocalStatInfo
{
 public:

  StatInfoList* statInfoList()
  {
    return _createOrGetStatInfoList();
  }
  void merge(const impl::LoopStatInfo& stat_info, const ForLoopTraceInfo& trace_info)
  {
    StatInfoList* stat_list = _createOrGetStatInfoList();
    stat_list->merge(stat_info, trace_info);
  }

 private:

  StatInfoList* _createOrGetStatInfoList()
  {
    if (!m_stat_info_list)
      m_stat_info_list = global_all_stat_info_list.createStatInfoList();
    return m_stat_info_list;
  }

 private:

  StatInfoList* m_stat_info_list = nullptr;
};
thread_local ThreadLocalStatInfo thread_local_stat_info;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StatInfoList* ProfilingRegistry::
threadLocalInstance()
{
  return thread_local_stat_info.statInfoList();
}

void ProfilingRegistry::
printExecutionStats(std::ostream& o)
{
  global_stat.printInfos(o);
  global_all_stat_info_list.print(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::LoopStatInfo::
printInfos(std::ostream& o)
{
  Int64 nb_loop_parallel_for = m_nb_loop_parallel_for;
  if (nb_loop_parallel_for == 0)
    return;
  Int64 nb_chunk_parallel_for = m_nb_chunk_parallel_for;
  Int64 total_time = m_total_time;
  double x = static_cast<double>(total_time);
  double x1 = 0.0;
  if (nb_loop_parallel_for > 0)
    x1 = x / static_cast<double>(nb_loop_parallel_for);
  double x2 = 0.0;
  if (nb_chunk_parallel_for > 0)
    x2 = x / static_cast<double>(nb_chunk_parallel_for);
  o << "LoopStat: global_time (ms) = " << x / 1.0e6 << "\n";
  o << "LoopStat: global_nb_loop   = " << std::setw(10) << nb_loop_parallel_for << " time=" << x1 << "\n";
  o << "LoopStat: global_nb_chunk  = " << std::setw(10) << nb_chunk_parallel_for << " time=" << x2 << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StatInfoList::
merge(const impl::LoopStatInfo& loop_stat_info, const ForLoopTraceInfo& loop_trace_info)
{
  global_stat.merge(loop_stat_info);
  String loop_name = "Unknown";
  if (loop_trace_info.isValid()) {
    loop_name = loop_trace_info.loopName();
    if (loop_name.empty())
      loop_name = loop_trace_info.traceInfo().name();
  }
  m_stat_map[loop_name].merge(loop_stat_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StatInfoList::
print(std::ostream& o)
{
  o << "TaskStat\n";
  o << std::setw(10) << "Nloop" << std::setw(10) << "Nchunk"
    << std::setw(10) << " T (us)" << std::setw(11) << "Tc (ns)\n";
  Int64 cumulative_total = 0;
  for (const auto& x : m_stat_map) {
    const impl::LoopStatInfo& s = x.second;
    Int64 nb_loop = s.m_nb_loop_parallel_for;
    Int64 nb_chunk = s.m_nb_chunk_parallel_for;
    Int64 total_time = s.m_total_time;
    Int64 time_per_chunk = (nb_chunk == 0) ? 0 : (total_time / nb_chunk);
    o << std::setw(10) << nb_loop << std::setw(10) << nb_chunk
      << std::setw(10) << total_time / 1000 << std::setw(10) << time_per_chunk << "  " << x.first << "\n";
    cumulative_total += total_time;
  }
  o << "TOTAL=" << cumulative_total / 1000000 << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
