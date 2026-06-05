// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerMng.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Variable Synchronizer Manager.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableSynchronizerMng.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/internal/MemoryBuffer.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IVariable.h"

#include <map>
#include <mutex>
#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Synchronization statistics.
 *
 * When the before/after synchronization comparison is active, each rank
 * knows for its part whether the compared values are the same or not.
 *
 * However, a reduction must be performed over all ranks to have
 * a global view of the comparison (because only one rank for which
 * the comparison is different is sufficient to consider the comparison to be
 * different).
 *
 * Since it is too costly to perform the reduction for every synchronization,
 * we maintain a list of comparisons and process this list when it
 * reaches a certain size or if it is explicitly requested.
 *
 */
class VariableSynchronizerStats
: public TraceAccessor
{
 public:

  // We use a ReduceMin for the comparison value.
  // For them to be considered identical, all ranks
  // must be identical. One 'Unknown' rank is enough to consider
  // it 'Unknown'. Therefore, 'Unknown' must be the lowest value and 'Same' the highest.
  static constexpr unsigned char LOCAL_UNKNOWN = 0;
  static constexpr unsigned char LOCAL_DIFF = 1;
  static constexpr unsigned char LOCAL_SAME = 2;

 public:

  class StatInfo
  {
   public:

    void add(const StatInfo& x)
    {
      m_count += x.m_count;
      m_nb_same += x.m_nb_same;
      m_nb_different += x.m_nb_different;
      m_nb_unknown += x.m_nb_unknown;
    }

   public:

    Int32 m_count = 0;
    Int32 m_nb_same = 0;
    Int32 m_nb_different = 0;
    Int32 m_nb_unknown = 0;
  };

 public:

  explicit VariableSynchronizerStats(VariableSynchronizerMng* vsm)
  : TraceAccessor(vsm->traceMng())
  , m_variable_synchronizer_mng(vsm)
  {}

 public:

  void init()
  {
    if (m_is_event_registered)
      ARCANE_FATAL("instance is already initialized.");
    auto handler = [&](const VariableSynchronizerEventArgs& args) {
      _handleEvent(args);
    };
    m_variable_synchronizer_mng->onSynchronized().attach(m_observer_pool, handler);
    m_is_event_registered = true;
  }

  void flushPendingStats(IParallelMng* pm);

  Int32 dumpStats(std::ostream& ostr)
  {
    std::streamsize old_precision = ostr.precision(20);
    ostr << "Synchronization Stats\n";
    ostr << Trace::Width(8) << "Total"
         << Trace::Width(8) << "  Nb "
         << Trace::Width(8) << "  Nb "
         << Trace::Width(8) << " Nb  "
         << "   Variable name"
         << "\n";
    ostr << Trace::Width(8) << "Count"
         << Trace::Width(8) << "Same"
         << Trace::Width(8) << "Diff"
         << Trace::Width(8) << "Unknown"
         << "\n";
    StatInfo total_stat;
    for (const auto& p : m_stats) {
      total_stat.add(p.second);
      ostr << " " << Trace::Width(7) << p.second.m_count
           << " " << Trace::Width(7) << p.second.m_nb_same
           << " " << Trace::Width(7) << p.second.m_nb_different
           << " " << Trace::Width(7) << p.second.m_nb_unknown
           << "   " << p.first
           << "\n";
    }
    ostr << "\n";
    ostr << " " << Trace::Width(7) << total_stat.m_count
         << " " << Trace::Width(7) << total_stat.m_nb_same
         << " " << Trace::Width(7) << total_stat.m_nb_different
         << " " << Trace::Width(7) << total_stat.m_nb_unknown
         << "   "
         << "TOTAL"
         << "\n\n";
    ostr.precision(old_precision);
    return total_stat.m_count;
  }

 private:

  VariableSynchronizerMng* m_variable_synchronizer_mng = nullptr;
  EventObserverPool m_observer_pool;
  std::map<String, StatInfo> m_stats;
  bool m_is_event_registered = false;
  UniqueArray<String> m_pending_variable_name_list;
  UniqueArray<unsigned char> m_pending_compare_status_list;

 private:

  void _handleEvent(const VariableSynchronizerEventArgs& args);
};

void VariableSynchronizerStats::
_handleEvent(const VariableSynchronizerEventArgs& args)
{
  // We only process end-of-synchronization events
  if (args.state() != VariableSynchronizerEventArgs::State::EndSynchronize)
    return;
  if (!m_variable_synchronizer_mng->isDoingStats())
    return;
  Int32 level = m_variable_synchronizer_mng->synchronizationCompareLevel();
  IParallelMng* pm = m_variable_synchronizer_mng->parallelMng();
  auto compare_status_list = args.compareStatusList();
  {
    Int32 index = 0;
    for (IVariable* var : args.variables()) {
      m_pending_variable_name_list.add(var->fullName());
      VariableSynchronizerEventArgs::CompareStatus s = compare_status_list[index];
      unsigned char rs = LOCAL_UNKNOWN; // Compare == Unknown;
      if (s == VariableSynchronizerEventArgs::CompareStatus::Same)
        rs = LOCAL_SAME;
      else if (s == VariableSynchronizerEventArgs::CompareStatus::Different)
        rs = LOCAL_DIFF;
      m_pending_compare_status_list.add(rs);
      ++index;
      if (level >= 2) {
        // We perform the reduction here because we want to know immediately if there is a
        // difference.
        unsigned char global_rs = pm->reduce(Parallel::ReduceMax, rs);
        if (global_rs == LOCAL_SAME) {
          info() << "Synchronize: same values for variable name=" << var->fullName();
          if (level >= 3)
            info() << "Stack=" << platform::getStackTrace();
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerStats::
flushPendingStats(IParallelMng* pm)
{
  Int32 nb_pending = m_pending_variable_name_list.size();
  Int32 total_nb_pending = pm->reduce(Parallel::ReduceMax, nb_pending);
  if (total_nb_pending != nb_pending)
    ARCANE_FATAL("Bad number of pending stats local={0} global={1}", nb_pending, total_nb_pending);
  pm->reduce(Parallel::ReduceMin, m_pending_compare_status_list);
  for (Int32 i = 0; i < total_nb_pending; ++i) {
    unsigned char rs = m_pending_compare_status_list[i];
    auto& v = m_stats[m_pending_variable_name_list[i]];
    if (rs == LOCAL_SAME)
      ++v.m_nb_same;
    else if (rs == LOCAL_DIFF)
      ++v.m_nb_different;
    else
      ++v.m_nb_unknown;
    ++v.m_count;
  }
  m_pending_variable_name_list.clear();
  m_pending_compare_status_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::
VariableSynchronizerMng(IVariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
, m_parallel_mng(vm->parallelMng())
, m_stats(new VariableSynchronizerStats(this))
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_AUTO_COMPARE_SYNCHRONIZE", true)) {
    m_synchronize_compare_level = v.value();
    // If comparison is active, statistics are also active
    m_is_doing_stats = m_synchronize_compare_level > 0;
  }
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_SYNCHRONIZE_STATS", true))
    m_is_doing_stats = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::
~VariableSynchronizerMng()
{
  delete m_stats;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
initialize()
{
  m_stats->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
dumpStats(std::ostream& ostr) const
{
  if (!m_parallel_mng->isParallel())
    return;
  {
    OStringStream ostr2;
    Int32 count = m_stats->dumpStats(ostr2());
    if (count > 0)
      ostr << ostr2.str();
  }
  m_internal_api.dumpStats(ostr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
flushPendingStats()
{
  if (isDoingStats())
    m_stats->flushPendingStats(m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages a pool of buffers associated with an allocator.
 *
 * The methods of this class are thread-safe.
 */
class VariableSynchronizerMng::InternalApi::BufferList
{
 public:

  using MemoryBufferMap = std::map<MemoryBuffer*, Ref<MemoryBuffer>>;
  using MapList = std::map<IMemoryAllocator*, MemoryBufferMap>;

  using FreeList = std::map<IMemoryAllocator*, std::stack<Ref<MemoryBuffer>>>;

 public:

  Ref<MemoryBuffer> createSynchronizeBuffer(IMemoryAllocator* allocator);
  void releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v);
  void dumpStats(std::ostream& ostr) const;

 private:

  //! List of buffers currently in use by allocator
  MapList m_used_map;

  //! List of free buffers by allocator
  FreeList m_free_map;

  //! Mutex to protect buffer creation/retrieval
  mutable std::mutex m_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Creates or retrieves a buffer.
 *
 * It is possible to create buffers with a null allocator. In this
 * case, the default allocator will be used and therefore for
 * a given MemoryBuffer, new_buffer.allocator() will not necessarily equal allocator.
 * You must always use \a allocator.
 */
Ref<MemoryBuffer> VariableSynchronizerMng::InternalApi::BufferList::
createSynchronizeBuffer(IMemoryAllocator* allocator)
{
  std::scoped_lock lock(m_mutex);

  auto& free_map = m_free_map;
  auto x = free_map.find(allocator);
  Ref<MemoryBuffer> new_buffer;
  // Checks if a buffer is available in \a free_map.
  if (x == free_map.end()) {
    // No buffer associated with this allocator, so we create one
    new_buffer = MemoryBuffer::create(allocator);
  }
  else {
    auto& buffer_stack = x->second;
    // If the stack is empty, we create a buffer. Otherwise, we take the first
    // from the stack.
    if (buffer_stack.empty()) {
      new_buffer = MemoryBuffer::create(allocator);
    }
    else {
      new_buffer = buffer_stack.top();
      buffer_stack.pop();
    }
  }

  // Registers the instance in the used list
  m_used_map[allocator].insert(std::make_pair(new_buffer.get(), new_buffer));
  return new_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::BufferList::
releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v)
{
  std::scoped_lock lock(m_mutex);

  auto& main_map = m_used_map;
  auto x = main_map.find(allocator);
  if (x == main_map.end())
    ARCANE_FATAL("Invalid allocator '{0}'", allocator);

  auto& sub_map = x->second;
  auto x2 = sub_map.find(v);
  if (x2 == sub_map.end())
    ARCANE_FATAL("Invalid buffer '{0}'", v);

  Ref<MemoryBuffer> ref_memory = x2->second;

  sub_map.erase(x2);

  m_free_map[allocator].push(ref_memory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::BufferList::
dumpStats(std::ostream& ostr) const
{
  std::scoped_lock lock(m_mutex);

  //! List of buffers currently in use by allocator
  for (const auto& x : m_used_map)
    ostr << "SynchronizeBuffer: nb_used_map = " << x.second.size() << "\n";

  //! List of free buffers by allocator
  for (const auto& x : m_free_map)
    ostr << "SynchronizeBuffer: nb_free_map = " << x.second.size() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::InternalApi::
InternalApi(VariableSynchronizerMng* vms)
: TraceAccessor(vms->traceMng())
, m_synchronizer_mng(vms)
, m_buffer_list(new BufferList())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::InternalApi::
~InternalApi()
{
  // The destructor cannot be deleted because 'm_buffer_list' is not
  // known when the class is defined.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<MemoryBuffer> VariableSynchronizerMng::InternalApi::
createSynchronizeBuffer(IMemoryAllocator* allocator)
{
  return m_buffer_list->createSynchronizeBuffer(allocator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::
releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v)
{
  m_buffer_list->releaseSynchronizeBuffer(allocator, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::
dumpStats(std::ostream& ostr) const
{
  m_buffer_list->dumpStats(ostr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
