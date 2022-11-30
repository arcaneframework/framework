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

#include "arcane/utils/internal/ProfilingInternal.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ForLoopStatInfoList::
ForLoopStatInfoList()
: m_p(new ForLoopStatInfoListImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ForLoopStatInfoList::
~ForLoopStatInfoList()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  impl::ForLoopCumulativeStat global_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ScopedStatLoop::
ScopedStatLoop(ForLoopOneExecStat* s)
: m_stat_info(s)
{
  if (m_stat_info) {
    m_begin_time = platform::getRealTimeNS();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ScopedStatLoop::
~ScopedStatLoop()
{
  if (m_stat_info) {
    Int64 end_time = platform::getRealTimeNS();
    m_stat_info->setBeginTime(m_begin_time);
    m_stat_info->setEndTime(end_time);
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

  void visit(const std::function<void(const impl::ForLoopStatInfoList&)>& f)
  {
    for (const auto& x : m_stat_info_list_vector)
      f(*x);
  }

  public :

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
visitLoopStat(const std::function<void(const impl::ForLoopStatInfoList&)>& f)
{
  global_all_stat_info_list.visit(f);
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

const impl::ForLoopCumulativeStat& ProfilingRegistry::
globalLoopStat()
{
  return global_stat;
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
