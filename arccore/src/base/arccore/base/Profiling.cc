// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiling.cc                                                (C) 2000-2025 */
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
