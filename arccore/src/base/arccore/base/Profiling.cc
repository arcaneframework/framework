// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiling.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Classes pour gérer le profilage.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Profiling.h"

#include "arccore/base/ForLoopTraceInfo.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/internal/ProfilingInternal.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>
#include <map>
#include <memory>
#include <set>

namespace
{
using namespace Arcane;

void _printGlobalLoopInfos(std::ostream& o, const Impl::ForLoopCumulativeStat& cumulative_stat)
{
  Int64 nb_loop_parallel_for = cumulative_stat.nbLoopParallelFor();
  if (nb_loop_parallel_for == 0)
    return;
  Int64 nb_chunk_parallel_for = cumulative_stat.nbChunkParallelFor();
  Int64 total_time = cumulative_stat.totalTime();
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

void _dumpOneLoopListStat(std::ostream& o, const Impl::ForLoopStatInfoList& stat_list)
{
  struct SortedStatInfo
  {
    bool operator<(const SortedStatInfo& rhs) const
    {
      return m_stat.execTime() > rhs.m_stat.execTime();
    }
    String m_name;
    Impl::ForLoopProfilingStat m_stat;
  };

  // Met 1 pour éviter de diviser par zéro.
  Int64 cumulative_total = 1;

  // Tri les fonctions par temps d'exécution décroissant
  std::set<SortedStatInfo> sorted_set;
  for (const auto& x : stat_list._internalImpl()->m_stat_map) {
    const auto& s = x.second;
    sorted_set.insert({ x.first, s });
    cumulative_total += s.execTime();
  }

  o << "ProfilingStat\n";
  o << std::setw(10) << "Ncall" << std::setw(10) << "Nchunk"
    << std::setw(11) << " T (ms)" << std::setw(10) << "Tck (ns)"
    << "     %  name\n";

  char old_filler = o.fill();
  for (const auto& x : sorted_set) {
    const Impl::ForLoopProfilingStat& s = x.m_stat;
    Int64 nb_loop = s.nbCall();
    Int64 nb_chunk = s.nbChunk();
    Int64 total_time_ns = s.execTime();
    Int64 total_time_us = total_time_ns / 1000;
    Int64 total_time_ms = total_time_us / 1000;
    Int64 total_time_remaining_us = total_time_us % 1000;
    Int64 time_per_chunk = (nb_chunk == 0) ? 0 : (total_time_ns / nb_chunk);
    Int64 per_mil = (total_time_ns * 1000) / cumulative_total;
    Int64 percent = per_mil / 10;
    Int64 percent_digit = per_mil % 10;

    o << std::setw(10) << nb_loop << std::setw(10) << nb_chunk
      << std::setw(7) << total_time_ms << ".";
    o << std::setfill('0') << std::setw(3) << total_time_remaining_us << std::setfill(old_filler);
    o << std::setw(10) << time_per_chunk
      << std::setw(4) << percent << "." << percent_digit << "  " << x.m_name << "\n";
  }
  o << "TOTAL=" << cumulative_total / 1000000 << "\n";
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::ForLoopStatInfoList::
ForLoopStatInfoList()
: m_p(new ForLoopStatInfoListImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::ForLoopStatInfoList::
~ForLoopStatInfoList()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  Impl::ForLoopCumulativeStat global_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::ScopedStatLoop::
ScopedStatLoop(ForLoopOneExecStat* s)
: m_stat_info(s)
{
  if (m_stat_info) {
    m_begin_time = Platform::getRealTimeNS();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::ScopedStatLoop::
~ScopedStatLoop()
{
  if (m_stat_info) {
    Int64 end_time = Platform::getRealTimeNS();
    m_stat_info->setBeginTime(m_begin_time);
    m_stat_info->setEndTime(end_time);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllStatInfoList
{
 public:

  Impl::ForLoopStatInfoList* createForLoopStatInfoList()
  {
    std::lock_guard<std::mutex> lk(m_mutex);
    std::unique_ptr<Impl::ForLoopStatInfoList> x(new Impl::ForLoopStatInfoList());
    auto* ptr = x.get();
    m_for_loop_stat_info_list_vector.push_back(std::move(x));
    return ptr;
  }
  Impl::AcceleratorStatInfoList* createAcceleratorStatInfoList()
  {
    std::lock_guard<std::mutex> lk(m_mutex);
    std::unique_ptr<Impl::AcceleratorStatInfoList> x(new Impl::AcceleratorStatInfoList());
    auto* ptr = x.get();
    m_accelerator_stat_info_list_vector.push_back(std::move(x));
    return ptr;
  }

  void visitForLoop(const std::function<void(const Impl::ForLoopStatInfoList&)>& f)
  {
    for (const auto& x : m_for_loop_stat_info_list_vector)
      f(*x);
  }

  void visitAccelerator(const std::function<void(const Impl::AcceleratorStatInfoList&)>& f)
  {
    for (const auto& x : m_accelerator_stat_info_list_vector)
      f(*x);
  }

 public:

  std::mutex m_mutex;
  std::vector<std::unique_ptr<Impl::ForLoopStatInfoList>> m_for_loop_stat_info_list_vector;
  std::vector<std::unique_ptr<Impl::AcceleratorStatInfoList>> m_accelerator_stat_info_list_vector;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllStatInfoList global_all_stat_info_list;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Permet de gérer une instance de ForLoopStatInfoList par thread pour éviter les verroux
class ThreadLocalStatInfo
{
 public:

  Impl::ForLoopStatInfoList* forLoopStatInfoList()
  {
    return _createOrGetForLoopStatInfoList();
  }
  Impl::AcceleratorStatInfoList* acceleratorStatInfoList()
  {
    return _createOrGetAcceleratorStatInfoList();
  }
  void merge(const ForLoopOneExecStat& stat_info, const ForLoopTraceInfo& trace_info)
  {
    Impl::ForLoopStatInfoList* stat_list = _createOrGetForLoopStatInfoList();
    stat_list->merge(stat_info, trace_info);
  }

 private:

  Impl::ForLoopStatInfoList* _createOrGetForLoopStatInfoList()
  {
    if (!m_for_loop_stat_info_list)
      m_for_loop_stat_info_list = global_all_stat_info_list.createForLoopStatInfoList();
    return m_for_loop_stat_info_list;
  }
  Impl::AcceleratorStatInfoList* _createOrGetAcceleratorStatInfoList()
  {
    if (!m_accelerator_stat_info_list)
      m_accelerator_stat_info_list = global_all_stat_info_list.createAcceleratorStatInfoList();
    return m_accelerator_stat_info_list;
  }

 private:

  Impl::ForLoopStatInfoList* m_for_loop_stat_info_list = nullptr;
  Impl::AcceleratorStatInfoList* m_accelerator_stat_info_list = nullptr;
};
thread_local ThreadLocalStatInfo thread_local_stat_info;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ProfilingRegistry::m_profiling_level = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::ForLoopStatInfoList* ProfilingRegistry::
threadLocalInstance()
{
  return thread_local_stat_info.forLoopStatInfoList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::ForLoopStatInfoList* ProfilingRegistry::
_threadLocalForLoopInstance()
{
  return thread_local_stat_info.forLoopStatInfoList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::AcceleratorStatInfoList* ProfilingRegistry::
_threadLocalAcceleratorInstance()
{
  return thread_local_stat_info.acceleratorStatInfoList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfilingRegistry::
visitLoopStat(const std::function<void(const Impl::ForLoopStatInfoList&)>& f)
{
  global_all_stat_info_list.visitForLoop(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfilingRegistry::
visitAcceleratorStat(const std::function<void(const Impl::AcceleratorStatInfoList&)>& f)
{
  global_all_stat_info_list.visitAccelerator(f);
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

const Impl::ForLoopCumulativeStat& ProfilingRegistry::
globalLoopStat()
{
  return global_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::ForLoopProfilingStat::
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

void Impl::ForLoopStatInfoList::
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

void Impl::AcceleratorStatInfoList::
print(std::ostream& o) const
{
  const auto& htod = memoryTransfer(eMemoryTransferType::HostToDevice);
  const auto& dtoh = memoryTransfer(eMemoryTransferType::DeviceToHost);
  o << "MemoryTransferSTATS: HTOD = " << htod.m_nb_byte << " (" << htod.m_nb_call << ")"
    << " DTOH = " << dtoh.m_nb_byte << " (" << dtoh.m_nb_call << ")";
  const auto& cpu_fault = memoryPageFault(eMemoryPageFaultType::Cpu);
  const auto& gpu_fault = memoryPageFault(eMemoryPageFaultType::Gpu);
  o << " PageFaultCPU = " << cpu_fault.m_nb_fault << " (" << cpu_fault.m_nb_call << ")"
    << " PageFaultGPU = " << gpu_fault.m_nb_fault << " (" << gpu_fault.m_nb_call << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::
dumpProfilingStatistics(std::ostream& o)
{
  // Affiche les informations de profiling sur \a o
  _printGlobalLoopInfos(o, ProfilingRegistry::globalLoopStat());
  {
    auto f = [&](const Impl::ForLoopStatInfoList& stat_list) {
      _dumpOneLoopListStat(o, stat_list);
    };
    ProfilingRegistry::visitLoopStat(f);
  }
  // Avant d'afficher le profiling accélérateur, il faudrait être certain
  // qu'il est désactivé. Normalement, c'est le cas si on utilise ArcaneMainBatch.
  {
    auto f = [&](const Impl::AcceleratorStatInfoList& stat_list) {
      stat_list.print(o);
    };
    ProfilingRegistry::visitAcceleratorStat(f);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
