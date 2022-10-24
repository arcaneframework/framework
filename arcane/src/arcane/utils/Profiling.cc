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
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

class ForLoopStatInfoList::Impl
{
 public:

  // TODO Utiliser un hash pour le map plutôt qu'une String pour accélérer les comparaisons
  std::map<String, impl::ForLoopProfilingStat> m_stat_map;
};

ForLoopStatInfoList::
ForLoopStatInfoList()
: m_p(new Impl())
{
}

ForLoopStatInfoList::
~ForLoopStatInfoList()
{
  delete m_p;
}

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ForLoopCumulativeStat
{
 public:

  void printInfos(std::ostream& o);
  void merge(const ForLoopOneExecStat& s)
  {
    ++m_nb_loop_parallel_for;
    m_nb_chunk_parallel_for += s.nbChunk();
    m_total_time += s.execTime();
  }

 public:

  std::atomic<Int64> m_nb_loop_parallel_for = 0;
  std::atomic<Int64> m_nb_chunk_parallel_for = 0;
  std::atomic<Int64> m_total_time = 0;
};

namespace
{
  ForLoopCumulativeStat global_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ScopedStatLoop::
ScopedStatLoop(ForLoopOneExecStat* s)
: m_stat_info(s)
{
  if (m_stat_info) {
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
    m_stat_info->setExecTime(v_as_int64);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllForLoopStatInfoList
{
 public:

  impl::ForLoopStatInfoList* createStatInfoList()
  {
    std::lock_guard<std::mutex> lk(m_mutex);
    std::unique_ptr<impl::ForLoopStatInfoList> x(new impl::ForLoopStatInfoList);
    impl::ForLoopStatInfoList* ptr = x.get();
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
  std::vector<std::unique_ptr<impl::ForLoopStatInfoList>> m_stat_info_list_vector;
};
AllForLoopStatInfoList global_all_stat_info_list;

// Permet de gérer une instance de ForLoopStatInfoList par thread pour éviter les verroux
class ThreadLocalStatInfo
{
 public:

  impl::ForLoopStatInfoList* statInfoList()
  {
    return _createOrGetStatInfoList();
  }
  void merge(const ForLoopOneExecStat& stat_info, const ForLoopTraceInfo& trace_info)
  {
    impl::ForLoopStatInfoList* stat_list = _createOrGetStatInfoList();
    stat_list->merge(stat_info, trace_info);
  }

 private:

  impl::ForLoopStatInfoList* _createOrGetStatInfoList()
  {
    if (!m_stat_info_list)
      m_stat_info_list = global_all_stat_info_list.createStatInfoList();
    return m_stat_info_list;
  }

 private:

  impl::ForLoopStatInfoList* m_stat_info_list = nullptr;
};
thread_local ThreadLocalStatInfo thread_local_stat_info;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ProfilingRegistry::m_profiling_level = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ForLoopStatInfoList* ProfilingRegistry::
threadLocalInstance()
{
  return thread_local_stat_info.statInfoList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfilingRegistry::
printExecutionStats(std::ostream& o)
{
  global_stat.printInfos(o);
  global_all_stat_info_list.print(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfilingRegistry::
setProfilingLevel(Int32 level)
{
  m_profiling_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ForLoopCumulativeStat::
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::ForLoopProfilingStat::
add(const ForLoopOneExecStat& s)
{
  ++m_nb_call;
  m_nb_chunk += s.nbChunk();
  m_exec_time += s.execTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::ForLoopStatInfoList::
merge(const ForLoopOneExecStat& loop_stat_info, const ForLoopTraceInfo& loop_trace_info)
{
  global_stat.merge(loop_stat_info);
  String loop_name = "Unknown";
  if (loop_trace_info.isValid()) {
    loop_name = loop_trace_info.loopName();
    if (loop_name.empty())
      loop_name = loop_trace_info.traceInfo().name();
  }
  m_p->m_stat_map[loop_name].add(loop_stat_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::ForLoopStatInfoList::
print(std::ostream& o)
{
  o << "ProfilingStat\n";
  o << std::setw(10) << "Ncall" << std::setw(10) << "Nchunk"
    << std::setw(10) << " T (us)" << std::setw(11) << "Tck (ns)\n";
  Int64 cumulative_total = 0;
  for (const auto& x : m_p->m_stat_map) {
    const auto& s = x.second;
    Int64 nb_loop = s.nbCall();
    Int64 nb_chunk = s.nbChunk();
    Int64 total_time = s.execTime();
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
