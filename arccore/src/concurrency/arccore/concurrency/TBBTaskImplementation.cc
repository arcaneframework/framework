// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TBBTaskImplementation.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Implementation of tasks using TBB (Intel Threads Building Blocks).        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/IFunctor.h"
#include "arccore/base/ForLoopRanges.h"
#include "arccore/base/IObservable.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FixedArray.h"
#include "arccore/base/Profiling.h"
#include "arccore/base/CheckedConvert.h"
#include "arccore/base/FixedArray.h"
#include "arccore/base/ForLoopRunInfo.h"
#include "arccore/base/internal/DependencyInjection.h"

#include "arccore/concurrency/IThreadImplementation.h"
#include "arccore/concurrency/Task.h"
#include "arccore/concurrency/ITaskImplementation.h"
#include "arccore/concurrency/TaskFactory.h"
#include "arccore/concurrency/ParallelFor.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"

#include <new>
#include <stack>
#include <vector>

// This macro must be defined for the class 'blocked_rangeNd' to be available

#define TBB_PREVIEW_BLOCKED_RANGE_ND 1

// The macro 'ARCCORE_USE_ONETBB' is defined in CMakeLists.txt
// if compiling with the OneTBB version 2021+
// (https://github.com/oneapi-src/oneTBB.git)
// Eventually, this will be the only version supported by Arcane.

// Necessary to access task_scheduler_handle
#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#include <tbb/tbb.h>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/global_control.h>

#include <thread>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class TBBTaskImplementation;

// TODO: use a specific memory pool to manage the
// OneTBBTask to optimize the new/delete of instances of this class.
// Previously, with older versions of TBB, this was managed with
// the method 'tbb::task::allocate_child()'.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if (TBB_VERSION_MAJOR > 2022) || (TBB_VERSION_MAJOR == 2022 && TBB_VERSION_MINOR > 0) || defined __TBB_blocked_nd_range_H

// The class "blocked_rangeNd" was removed in version
// 2022.0.0 and replaced by "blocked_nd_range".
template <typename Value, unsigned int N>
using blocked_nd_range = tbb::blocked_nd_range<Value, N>;

#else

template <typename Value, unsigned int N>
using blocked_nd_range = tbb::blocked_rangeNd<Value, N>;

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  constexpr Int32 cache_line_size = 64;
  // Positive if execution statistics are retrieved
  bool isStatActive()
  {
    return ProfilingRegistry::hasProfiling();
  }

  /*!
 * \brief Class that ensures execution statistics are recorded
 * even in case of an exception.
 */
  class ScopedExecInfo
  {
   public:

    explicit ScopedExecInfo(const ForLoopRunInfo& run_info)
    : m_run_info(run_info)
    {
      // If run_info.execInfo() is not null, we use it.
      // This means that the caller will manage the execution statistics
      // execution statistics. Otherwise, we use m_stat_info if execution statistics
      // are requested.
      ForLoopOneExecStat* ptr = run_info.execStat();
      if (ptr) {
        m_stat_info_ptr = ptr;
        m_use_own_run_info = false;
      }
      else
        m_stat_info_ptr = isStatActive() ? &m_stat_info : nullptr;
    }
    ~ScopedExecInfo()
    {
#ifdef PRINT_STAT_INFO
      if (m_stat_info_ptr) {
        bool is_valid = m_run_info.traceInfo().isValid();
        if (!is_valid)
          std::cout << "ADD_OWN_RUN_INFO nb_chunk=" << m_stat_info_ptr->nbChunk()
                    << " stack=" << platform::getStackTrace()
                    << "\n";
        else
          std::cout << "ADD_OWN_RUN_INFO nb_chunk=" << m_stat_info_ptr->nbChunk()
                    << " trace_name=" << m_run_info.traceInfo().traceInfo().name() << "\n";
      }
#endif
      if (m_stat_info_ptr && m_use_own_run_info) {
        ProfilingRegistry::_threadLocalForLoopInstance()->merge(*m_stat_info_ptr, m_run_info.traceInfo());
      }
    }

   public:

    ForLoopOneExecStat* statInfo() const { return m_stat_info_ptr; }
    bool isOwn() const { return m_use_own_run_info; }

   private:

    ForLoopOneExecStat m_stat_info;
    ForLoopOneExecStat* m_stat_info_ptr = nullptr;
    ForLoopRunInfo m_run_info;
    //! Indicates if m_stat_info is used
    bool m_use_own_run_info = true;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  inline int _currentTaskTreadIndex()
  {
    // NOTE: With OneTBB 2021, the value is no longer '0' if this method is called
    // from a thread outside of a task_arena. With version 2021,
    // the value is 65535.
    // NOTE: It seems this is a bug in 2021.3.
    return tbb::this_task_arena::current_thread_index();
  }

  inline blocked_nd_range<Int32, 1>
  _toTBBRange(const ComplexForLoopRanges<1>& r)
  {
    return { { r.lowerBound<0>(), r.upperBound<0>() } };
  }

  inline blocked_nd_range<Int32, 2>
  _toTBBRange(const ComplexForLoopRanges<2>& r)
  {
    return { { r.lowerBound<0>(), r.upperBound<0>() },
             { r.lowerBound<1>(), r.upperBound<1>() } };
  }

  inline blocked_nd_range<Int32, 3>
  _toTBBRange(const ComplexForLoopRanges<3>& r)
  {
    return { { r.lowerBound<0>(), r.upperBound<0>() },
             { r.lowerBound<1>(), r.upperBound<1>() },
             { r.lowerBound<2>(), r.upperBound<2>() } };
  }

  inline blocked_nd_range<Int32, 4>
  _toTBBRange(const ComplexForLoopRanges<4>& r)
  {
    return { { r.lowerBound<0>(), r.upperBound<0>() },
             { r.lowerBound<1>(), r.upperBound<1>() },
             { r.lowerBound<2>(), r.upperBound<2>() },
             { r.lowerBound<3>(), r.upperBound<3>() } };
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  inline blocked_nd_range<Int32, 2>
  _toTBBRangeWithGrain(const blocked_nd_range<Int32, 2>& r, FixedArray<size_t, 2> grain_sizes)
  {
    return { { r.dim(0).begin(), r.dim(0).end(), grain_sizes[0] },
             { r.dim(1).begin(), r.dim(1).end(), grain_sizes[1] } };
  }

  inline blocked_nd_range<Int32, 3>
  _toTBBRangeWithGrain(const blocked_nd_range<Int32, 3>& r, FixedArray<size_t, 3> grain_sizes)
  {
    return { { r.dim(0).begin(), r.dim(0).end(), grain_sizes[0] },
             { r.dim(1).begin(), r.dim(1).end(), grain_sizes[1] },
             { r.dim(2).begin(), r.dim(2).end(), grain_sizes[2] } };
  }

  inline blocked_nd_range<Int32, 4>
  _toTBBRangeWithGrain(const blocked_nd_range<Int32, 4>& r, FixedArray<size_t, 4> grain_sizes)
  {
    return { { r.dim(0).begin(), r.dim(0).end(), grain_sizes[0] },
             { r.dim(1).begin(), r.dim(1).end(), grain_sizes[1] },
             { r.dim(2).begin(), r.dim(2).end(), grain_sizes[2] },
             { r.dim(3).begin(), r.dim(3).end(), grain_sizes[3] } };
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  inline ComplexForLoopRanges<2>
  _fromTBBRange(const blocked_nd_range<Int32, 2>& r)
  {
    using BoundsType = ArrayBounds<MDDim2>;
    using ArrayExtentType = BoundsType::ArrayExtentType;

    BoundsType lower_bounds(ArrayExtentType(r.dim(0).begin(), r.dim(1).begin()));
    auto s0 = static_cast<Int32>(r.dim(0).size());
    auto s1 = static_cast<Int32>(r.dim(1).size());
    BoundsType sizes(ArrayExtentType(s0, s1));
    return { lower_bounds, sizes };
  }

  inline ComplexForLoopRanges<3>
  _fromTBBRange(const blocked_nd_range<Int32, 3>& r)
  {
    using BoundsType = ArrayBounds<MDDim3>;
    using ArrayExtentType = BoundsType::ArrayExtentType;

    BoundsType lower_bounds(ArrayExtentType(r.dim(0).begin(), r.dim(1).begin(), r.dim(2).begin()));
    auto s0 = static_cast<Int32>(r.dim(0).size());
    auto s1 = static_cast<Int32>(r.dim(1).size());
    auto s2 = static_cast<Int32>(r.dim(2).size());
    BoundsType sizes(ArrayExtentType(s0, s1, s2));
    return { lower_bounds, sizes };
  }

  inline ComplexForLoopRanges<4>
  _fromTBBRange(const blocked_nd_range<Int32, 4>& r)
  {
    using BoundsType = ArrayBounds<MDDim4>;
    using ArrayExtentType = typename BoundsType::ArrayExtentType;

    BoundsType lower_bounds(ArrayExtentType(r.dim(0).begin(), r.dim(1).begin(), r.dim(2).begin(), r.dim(3).begin()));
    auto s0 = static_cast<Int32>(r.dim(0).size());
    auto s1 = static_cast<Int32>(r.dim(1).size());
    auto s2 = static_cast<Int32>(r.dim(2).size());
    auto s3 = static_cast<Int32>(r.dim(3).size());
    BoundsType sizes(ArrayExtentType(s0, s1, s2, s3));
    return { lower_bounds, sizes };
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneTBBTaskFunctor
{
 public:

  OneTBBTaskFunctor(ITaskFunctor* functor, ITask* task)
  : m_functor(functor)
  , m_task(task)
  {}

 public:

  void operator()() const
  {
    if (m_functor) {
      ITaskFunctor* tf = m_functor;
      m_functor = nullptr;
      TaskContext task_context(m_task);
      //cerr << "FUNC=" << typeid(*tf).name();
      tf->executeFunctor(task_context);
    }
  }

 public:

  mutable ITaskFunctor* m_functor;
  ITask* m_task;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneTBBTask
: public ITask
{
 public:

  static const int FUNCTOR_CLASS_SIZE = 32;

 public:

  explicit OneTBBTask(ITaskFunctor* f)
  : m_functor(f)
  {
    m_functor = f->clone(m_functor_buf.data(), FUNCTOR_CLASS_SIZE);
  }

 public:

  OneTBBTaskFunctor taskFunctor() { return OneTBBTaskFunctor(m_functor, this); }
  void launchAndWait() override;
  void launchAndWait(ConstArrayView<ITask*> tasks) override;

 protected:

  ITask* _createChildTask(ITaskFunctor* functor) override;

 public:

  ITaskFunctor* m_functor = nullptr;
  FixedArray<char, FUNCTOR_CLASS_SIZE> m_functor_buf;
};
using TBBTask = OneTBBTask;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Do not use the local observer on the task_arena.
 * Use the global observer on the scheduler.
 * For the ID, use tbb::this_task_arena::current_thread_index().
 */
class TBBTaskImplementation
: public ITaskImplementation
{
  class Impl;
  class ParallelForExecute;
  template <int RankValue>
  class MDParallelForExecute;

 public:

  // For performance reasons, aligns to a cache line
  // and uses padding.
  class ARCCORE_ALIGNAS_PACKED(64) TaskThreadInfo
  {
   public:

    TaskThreadInfo()
    : m_task_index(-1)
    {}

   public:

    void setTaskIndex(Integer v) { m_task_index = v; }
    Integer taskIndex() const { return m_task_index; }

   private:

    Integer m_task_index;
  };

  /*!
   * \brief Class for positioning TaskThreadInfo::taskIndex().
   *
   * Allows positioning the value of TaskThreadInfo::taskIndex()
   * during construction and restoring the previous value
   * in the destructor.
   */
  class TaskInfoLockGuard
  {
   public:

    TaskInfoLockGuard(TaskThreadInfo* tti, Integer task_index)
    : m_tti(tti)
    , m_old_task_index(-1)
    {
      if (tti) {
        m_old_task_index = tti->taskIndex();
        tti->setTaskIndex(task_index);
      }
    }
    ~TaskInfoLockGuard()
    {
      if (m_tti)
        m_tti->setTaskIndex(m_old_task_index);
    }

   private:

    TaskThreadInfo* m_tti;
    Integer m_old_task_index;
  };

 public:

  TBBTaskImplementation() = default;
  ~TBBTaskImplementation() override;

 public:

  void build() {}
  void initialize(Int32 nb_thread) override;
  void terminate() override;

  ITask* createRootTask(ITaskFunctor* f) override
  {
    OneTBBTask* t = new OneTBBTask(f);
    return t;
  }

  void executeParallelFor(Int32 begin, Int32 size, const ParallelLoopOptions& options, IRangeFunctor* f) final;
  void executeParallelFor(Int32 begin, Int32 size, Integer grain_size, IRangeFunctor* f) final;
  void executeParallelFor(Int32 begin, Int32 size, IRangeFunctor* f) final
  {
    executeParallelFor(begin, size, TaskFactory::defaultParallelLoopOptions(), f);
  }
  void executeParallelFor(const ParallelFor1DLoopInfo& loop_info) override;

  void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<1>* functor) final
  {
    _executeMDParallelFor<1>(loop_ranges, functor, run_info);
  }
  void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<2>* functor) final
  {
    _executeMDParallelFor<2>(loop_ranges, functor, run_info);
  }
  void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<3>* functor) final
  {
    _executeMDParallelFor<3>(loop_ranges, functor, run_info);
  }
  void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<4>* functor) final
  {
    _executeMDParallelFor<4>(loop_ranges, functor, run_info);
  }

  bool isActive() const final
  {
    return m_is_active;
  }

  Int32 currentTaskThreadIndex() const final
  {
    return (nbAllowedThread() <= 1) ? 0 : _currentTaskTreadIndex();
  }

  Int32 currentTaskIndex() const final;

  void printInfos(std::ostream& o) const final;

 public:

  /*!
   * \brief Instance of \a TaskThreadInfo associated with the current thread.
   *
   * May be null if the current thread is not associated with a TBB thread
   * or if it is outside the execution of a task or a parallel loop.
   */
  TaskThreadInfo* currentTaskThreadInfo() const;

 private:

  bool m_is_active = false;
  Impl* m_p = nullptr;

 private:

  template <int RankValue> void
  _executeMDParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                        IMDRangeFunctor<RankValue>* functor,
                        const ForLoopRunInfo& run_info);
  void _executeParallelFor(const ParallelFor1DLoopInfo& loop_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBTaskImplementation::Impl
{
  class TaskObserver
  : public tbb::task_scheduler_observer
  {
   public:

    explicit TaskObserver(TBBTaskImplementation::Impl* p)
    : tbb::task_scheduler_observer(p->m_main_arena)
    , m_p(p)
    {
    }
    void on_scheduler_entry(bool is_worker) override
    {
      m_p->notifyThreadCreated(is_worker);
    }
    void on_scheduler_exit(bool is_worker) override
    {
      m_p->notifyThreadDestroyed(is_worker);
    }
    TBBTaskImplementation::Impl* m_p;
  };

 public:

  Impl()
  : m_task_observer(this)
  , m_thread_task_infos(cache_line_size)
  {
    m_nb_allowed_thread = tbb::info::default_concurrency();
    _init();
  }
  Impl(Int32 nb_thread)
  : m_main_arena(nb_thread)
  , m_task_observer(this)
  , m_thread_task_infos(cache_line_size)
  {
    m_nb_allowed_thread = nb_thread;
    _init();
  }

 public:

  Int32 nbAllowedThread() const { return m_nb_allowed_thread; }
  TaskThreadInfo* threadTaskInfo(Integer index) { return &m_thread_task_infos[index]; }

 private:

  Int32 m_nb_allowed_thread = 0;

 public:

  void terminate()
  {
    for (auto x : m_sub_arena_list) {
      if (x)
        x->terminate();
      delete x;
    }
    m_sub_arena_list.clear();
    m_main_arena.terminate();
    m_task_observer.observe(false);
    oneapi::tbb::finalize(m_task_scheduler_handle);
  }

 public:

  void notifyThreadCreated(bool is_worker)
  {
    std::thread::id my_thread_id = std::this_thread::get_id();

    // With OneTBB, this method is called every time we enter
    // our 'task_arena'. Since the notification method should only be called once,
    // we use a set to keep track of the threads already created.
    // NOTE: This method cannot be used with the historical TBB version
    // (2018) because this 'contains' method does not exist
    if (m_constructed_thread_map.contains(my_thread_id))
      return;
    m_constructed_thread_map.insert(my_thread_id);

    {
      if (TaskFactory::verboseLevel() >= 1) {
        std::ostringstream ostr;
        ostr << "TBB: CREATE THREAD"
             << " nb_allowed=" << m_nb_allowed_thread
             << " tbb_default_allowed=" << tbb::info::default_concurrency()
             << " id=" << my_thread_id
             << " arena_id=" << _currentTaskTreadIndex()
             << " is_worker=" << is_worker
             << "\n";
        std::cout << ostr.str();
      }
      TaskFactoryInternal::notifyThreadCreated();
    }
  }

  void notifyThreadDestroyed([[maybe_unused]] bool is_worker)
  {
    // With OneTBB, this method is called every time we exit
    // the main arena. Therefore, it does not truly correspond to a
    // thread destruction. So we do nothing for this notification.
    // TODO: Look into how we can be notified of the actual thread destruction.
  }

 private:

#if TBB_VERSION_MAJOR > 2021 || (TBB_VERSION_MAJOR == 2021 && TBB_VERSION_MINOR > 5)
  oneapi::tbb::task_scheduler_handle m_task_scheduler_handle = oneapi::tbb::attach();
#else
  oneapi::tbb::task_scheduler_handle m_task_scheduler_handle = tbb::task_scheduler_handle::get();
#endif

 public:

  tbb::task_arena m_main_arena;
  //! Array whose i-th element contains the tbb::task_arena for \a i thread.
  std::vector<tbb::task_arena*> m_sub_arena_list;

 private:

  TaskObserver m_task_observer;
  std::mutex m_thread_created_mutex;
  std::vector<TaskThreadInfo> m_thread_task_infos;
  tbb::concurrent_set<std::thread::id> m_constructed_thread_map;
  void _init()
  {
    ConcurrencyBase::_setMaxAllowedThread(m_nb_allowed_thread);

    if (TaskFactory::verboseLevel() >= 1) {
      std::cout << "TBB: TBBTaskImplementationInit nb_allowed_thread=" << m_nb_allowed_thread
                << " id=" << std::this_thread::get_id()
                << " version=" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR
                << "\n";
    }
    m_thread_task_infos.resize(m_nb_allowed_thread);
    m_task_observer.observe(true);
    Integer max_arena_size = m_nb_allowed_thread;
    // Artificially limit the number of tbb::task_arena
    // to avoid having too many allocated objects.
    if (max_arena_size > 512)
      max_arena_size = 512;
    if (max_arena_size < 2)
      max_arena_size = 2;
    m_sub_arena_list.resize(max_arena_size);
    m_sub_arena_list[0] = m_sub_arena_list[1] = nullptr;
    for (Integer i = 2; i < max_arena_size; ++i)
      m_sub_arena_list[i] = new tbb::task_arena(i);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Executor for a 1D loop.
 */
class TBBParallelFor
{
 public:

  TBBParallelFor(IRangeFunctor* f, Int32 nb_allowed_thread, ForLoopOneExecStat* stat_info)
  : m_functor(f)
  , m_stat_info(stat_info)
  , m_nb_allowed_thread(nb_allowed_thread)
  {}

 public:

  void operator()(tbb::blocked_range<Integer>& range) const
  {
#ifdef ARCCORE_CHECK
    if (TaskFactory::verboseLevel() >= 3) {
      std::ostringstream o;
      o << "TBB: INDEX=" << TaskFactory::currentTaskThreadIndex()
        << " id=" << std::this_thread::get_id()
        << " max_allowed=" << m_nb_allowed_thread
        << " range_begin=" << range.begin() << " range_size=" << range.size()
        << "\n";
      std::cout << o.str();
      std::cout.flush();
    }

    int tbb_index = _currentTaskTreadIndex();
    if (tbb_index < 0 || tbb_index >= m_nb_allowed_thread)
      ARCCORE_FATAL("Invalid index for thread idx={0} valid_interval=[0..{1}[",
                    tbb_index, m_nb_allowed_thread);
#endif

    if (m_stat_info)
      m_stat_info->incrementNbChunk();
    m_functor->executeFunctor(range.begin(), CheckedConvert::toInteger(range.size()));
  }

 private:

  IRangeFunctor* m_functor;
  ForLoopOneExecStat* m_stat_info = nullptr;
  Int32 m_nb_allowed_thread;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Executor for a multi-dimensional loop.
 */
template <int RankValue>
class TBBMDParallelFor
{
 public:

  TBBMDParallelFor(IMDRangeFunctor<RankValue>* f, Int32 nb_allowed_thread, ForLoopOneExecStat* stat_info)
  : m_functor(f)
  , m_stat_info(stat_info)
  , m_nb_allowed_thread(nb_allowed_thread)
  {}

 public:

  void operator()(blocked_nd_range<Int32, RankValue>& range) const
  {
#ifdef ARCCORE_CHECK
    if (TaskFactory::verboseLevel() >= 3) {
      std::ostringstream o;
      o << "TBB: INDEX=" << TaskFactory::currentTaskThreadIndex()
        << " id=" << std::this_thread::get_id()
        << " max_allowed=" << m_nb_allowed_thread
        << " MDFor ";
      for (Int32 i = 0; i < RankValue; ++i) {
        auto r0 = static_cast<Int32>(range.dim(i).begin());
        auto r1 = static_cast<Int32>(range.dim(i).size());
        o << " range" << i << " (begin=" << r0 << " size=" << r1 << ")";
      }
      o << "\n";
      std::cout << o.str();
      std::cout.flush();
    }

    int tbb_index = _currentTaskTreadIndex();
    if (tbb_index < 0 || tbb_index >= m_nb_allowed_thread)
      ARCCORE_FATAL("Invalid index for thread idx={0} valid_interval=[0..{1}[",
                    tbb_index, m_nb_allowed_thread);
#endif

    if (m_stat_info)
      m_stat_info->incrementNbChunk();
    m_functor->executeFunctor(_fromTBBRange(range));
  }

 private:

  IMDRangeFunctor<RankValue>* m_functor = nullptr;
  ForLoopOneExecStat* m_stat_info = nullptr;
  Int32 m_nb_allowed_thread;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Deterministic implementation of ParallelFor.
 *
 * The implementation is deterministic in the sense that it only depends on
 * the iteration interval (m_begin_index and m_size),
 * the specified number of threads (\a m_nb_thread), and the grain size
 * (\a m_grain_size).
 *
 * The algorithm used is similar to the one used by OpenMP for a
 * parallel for with the static option: the iteration interval
 * is divided into several blocks and each block is assigned to a task based
 * on a round-robin algorithm.
 * To determine the number of blocks, two cases are possible:
 * - if \a m_grain_size is not specified, the iteration interval
 * is divided into a number of blocks equal to the number of threads used.
 * - if \a m_grain_size is specified, the number of blocks will be equal
 * to \a m_size divided by \a m_grain_size.
 */
class TBBDeterministicParallelFor
{
 public:

  TBBDeterministicParallelFor(TBBTaskImplementation* impl, const TBBParallelFor& tbb_for,
                              Integer begin_index, Integer size, Integer grain_size, Integer nb_thread)
  : m_impl(impl)
  , m_tbb_for(tbb_for)
  , m_nb_thread(nb_thread)
  , m_begin_index(begin_index)
  , m_size(size)
  , m_grain_size(grain_size)
  , m_nb_block(0)
  , m_block_size(0)
  , m_nb_block_per_thread(0)
  {
    if (m_nb_thread < 1)
      m_nb_thread = 1;

    if (m_grain_size > 0) {
      m_block_size = m_grain_size;
      if (m_block_size > 0) {
        m_nb_block = m_size / m_block_size;
        if ((m_size % m_block_size) != 0)
          ++m_nb_block;
      }
      else
        m_nb_block = 1;
      m_nb_block_per_thread = m_nb_block / m_nb_thread;
      if ((m_nb_block % m_nb_thread) != 0)
        ++m_nb_block_per_thread;
    }
    else {
      if (m_nb_block < 1)
        m_nb_block = m_nb_thread;
      m_block_size = m_size / m_nb_block;
      m_nb_block_per_thread = 1;
    }
    if (TaskFactory::verboseLevel() >= 2) {
      std::cout << "TBBDeterministicParallelFor: BEGIN=" << m_begin_index << " size=" << m_size
                << " grain_size=" << m_grain_size
                << " nb_block=" << m_nb_block << " nb_thread=" << m_nb_thread
                << " nb_block_per_thread=" << m_nb_block_per_thread
                << " block_size=" << m_block_size
                << " block_size*nb_block=" << m_block_size * m_nb_block << '\n';
    }
  }

 public:

  /*!
   * \brief Operator for a given thread.
   *
   * Generally, range.size() will be one, because a thread will only
   * process one iteration, but this is not guaranteed by TBB.
   */
  void operator()(tbb::blocked_range<Integer>& range) const
  {
    auto nb_iter = static_cast<Integer>(range.size());
    for (Integer i = 0; i < nb_iter; ++i) {
      Integer task_id = range.begin() + i;
      for (Integer k = 0, kn = m_nb_block_per_thread; k < kn; ++k) {
        Integer block_id = task_id + (k * m_nb_thread);
        if (block_id < m_nb_block)
          _doBlock(task_id, block_id);
      }
    }
  }

  void _doBlock(Integer task_id, Integer block_id) const
  {
    TBBTaskImplementation::TaskInfoLockGuard guard(m_impl->currentTaskThreadInfo(), task_id);

    Integer iter_begin = block_id * m_block_size;
    Integer iter_size = m_block_size;
    if ((block_id + 1) == m_nb_block) {
      // For the last block, the size is the number of remaining elements
      iter_size = m_size - iter_begin;
    }
    iter_begin += m_begin_index;
#ifdef ARCCORE_CHECK
    if (TaskFactory::verboseLevel() >= 3) {
      std::ostringstream o;
      o << "TBB: DoBlock: BLOCK task_id=" << task_id << " block_id=" << block_id
        << " iter_begin=" << iter_begin << " iter_size=" << iter_size << '\n';
      std::cout << o.str();
      std::cout.flush();
    }
#endif
    if (iter_size > 0) {
      auto r = tbb::blocked_range<int>(iter_begin, iter_begin + iter_size);
      m_tbb_for(r);
    }
  }

 private:

  TBBTaskImplementation* m_impl;
  const TBBParallelFor& m_tbb_for;
  Integer m_nb_thread;
  Integer m_begin_index;
  Integer m_size;
  Integer m_grain_size;
  Integer m_nb_block;
  Integer m_block_size;
  Integer m_nb_block_per_thread;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBTaskImplementation::ParallelForExecute
{
 public:

  ParallelForExecute(TBBTaskImplementation* impl, const ParallelLoopOptions& options,
                     Integer begin, Integer size, IRangeFunctor* f, ForLoopOneExecStat* stat_info)
  : m_impl(impl)
  , m_begin(begin)
  , m_size(size)
  , m_functor(f)
  , m_options(options)
  , m_stat_info(stat_info)
  {}

 public:

  void operator()() const
  {
    Integer nb_thread = m_options.maxThread();
    TBBParallelFor pf(m_functor, nb_thread, m_stat_info);
    Integer gsize = m_options.grainSize();
    tbb::blocked_range<Integer> range(m_begin, m_begin + m_size);
    if (TaskFactory::verboseLevel() >= 1)
      std::cout << "TBB: TBBTaskImplementationInit ParallelForExecute begin=" << m_begin
                << " size=" << m_size << " gsize=" << gsize
                << " partitioner=" << (int)m_options.partitioner()
                << " nb_thread=" << nb_thread
                << " has_stat_info=" << (m_stat_info != nullptr)
                << '\n';

    if (gsize > 0)
      range = tbb::blocked_range<Integer>(m_begin, m_begin + m_size, gsize);

    if (m_options.partitioner() == ParallelLoopOptions::Partitioner::Static) {
      tbb::parallel_for(range, pf, tbb::static_partitioner());
    }
    else if (m_options.partitioner() == ParallelLoopOptions::Partitioner::Deterministic) {
      tbb::blocked_range<Integer> range2(0, nb_thread, 1);
      TBBDeterministicParallelFor dpf(m_impl, pf, m_begin, m_size, gsize, nb_thread);
      tbb::parallel_for(range2, dpf);
    }
    else
      tbb::parallel_for(range, pf);
  }

 private:

  TBBTaskImplementation* m_impl = nullptr;
  Integer m_begin;
  Integer m_size;
  IRangeFunctor* m_functor = nullptr;
  ParallelLoopOptions m_options;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int RankValue>
class TBBTaskImplementation::MDParallelForExecute
{
 public:

  MDParallelForExecute(TBBTaskImplementation* impl,
                       const ParallelLoopOptions& options,
                       const ComplexForLoopRanges<RankValue>& range,
                       IMDRangeFunctor<RankValue>* f, [[maybe_unused]] ForLoopOneExecStat* stat_info)
  : m_impl(impl)
  , m_tbb_range(_toTBBRange(range))
  , m_functor(f)
  , m_options(options)
  , m_stat_info(stat_info)
  {
    // We cannot modify the values of a tbb::blocked_rangeNd instance.
    // We must therefore reconstruct it completely.
    FixedArray<size_t, RankValue> all_grain_sizes;
    Int32 gsize = m_options.grainSize();
    if (gsize > 0) {
      // If the grain size is not zero, it must be distributed
      // across all dimensions. We start with the last one.
      // TODO: check why performance is sometimes
      // lower than what we get using a static partitioner.
      constexpr bool is_verbose = false;
      std::array<Int32, RankValue> range_extents = range.extents().asStdArray();
      double ratio = static_cast<double>(gsize) / static_cast<double>(range.nbElement());
      if constexpr (is_verbose) {
        std::cout << "GSIZE=" << gsize << " rank=" << RankValue << " ratio=" << ratio;
        for (Int32 i = 0; i < RankValue; ++i)
          std::cout << " range" << i << "=" << range_extents[i];
        std::cout << "\n";
      }
      Int32 index = RankValue - 1;
      Int32 remaining_grain = gsize;
      for (; index >= 0; --index) {
        Int32 current = range_extents[index];
        if constexpr (is_verbose)
          std::cout << "Check index=" << index << " remaining=" << remaining_grain << " current=" << current << "\n";
        if (remaining_grain > current) {
          all_grain_sizes[index] = current;
          remaining_grain /= current;
        }
        else {
          all_grain_sizes[index] = remaining_grain;
          break;
        }
      }
      for (Int32 i = 0; i < index; ++i)
        all_grain_sizes[i] = 1;
      if constexpr (is_verbose) {
        for (Int32 i = 0; i < RankValue; ++i)
          std::cout << " grain" << i << "=" << all_grain_sizes[i];
        std::cout << "\n";
      }
      m_tbb_range = _toTBBRangeWithGrain(m_tbb_range, all_grain_sizes);
    }
  }

 public:

  void operator()() const
  {
    Integer nb_thread = m_options.maxThread();
    TBBMDParallelFor<RankValue> pf(m_functor, nb_thread, m_stat_info);

    if (m_options.partitioner() == ParallelLoopOptions::Partitioner::Static) {
      tbb::parallel_for(m_tbb_range, pf, tbb::static_partitioner());
    }
    else if (m_options.partitioner() == ParallelLoopOptions::Partitioner::Deterministic) {
      // TODO: implement deterministic mode
      ARCCORE_THROW(NotImplementedException, "ParallelLoopOptions::Partitioner::Deterministic for multi-dimensionnal loops");
      //tbb::blocked_range<Integer> range2(0,nb_thread,1);
      //TBBDeterministicParallelFor dpf(m_impl,pf,m_begin,m_size,gsize,nb_thread);
      //tbb::parallel_for(range2,dpf);
    }
    else {
      tbb::parallel_for(m_tbb_range, pf);
    }
  }

 private:

  TBBTaskImplementation* m_impl = nullptr;
  blocked_nd_range<Int32, RankValue> m_tbb_range;
  IMDRangeFunctor<RankValue>* m_functor = nullptr;
  ParallelLoopOptions m_options;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TBBTaskImplementation::
~TBBTaskImplementation()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
initialize(Int32 nb_thread)
{
  if (nb_thread < 0)
    nb_thread = 0;
  m_is_active = (nb_thread != 1);
  if (nb_thread != 0)
    m_p = new Impl(nb_thread);
  else
    m_p = new Impl();
  ParallelLoopOptions opts = TaskFactory::defaultParallelLoopOptions();
  opts.setMaxThread(nbAllowedThread());
  TaskFactory::setDefaultParallelLoopOptions(opts);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
terminate()
{
  m_p->terminate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
printInfos(std::ostream& o) const
{
  o << "OneTBBTaskImplementation"
    << " version=" << TBB_VERSION_STRING
    << " interface=" << TBB_INTERFACE_VERSION
    << " runtime_interface=" << TBB_runtime_interface_version();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
_executeParallelFor(const ParallelFor1DLoopInfo& loop_info)
{
  ScopedExecInfo sei(loop_info.runInfo());
  ForLoopOneExecStat* stat_info = sei.statInfo();
  ::Arcane::Impl::ScopedStatLoop scoped_loop(sei.isOwn() ? stat_info : nullptr);

  Int32 begin = loop_info.beginIndex();
  Int32 size = loop_info.size();
  ParallelLoopOptions options = loop_info.runInfo().options().value_or(TaskFactory::defaultParallelLoopOptions());
  IRangeFunctor* f = loop_info.functor();
  ARCCORE_CHECK_POINTER(f);

  Integer max_thread = options.maxThread();
  Integer nb_allowed_thread = m_p->nbAllowedThread();
  if (max_thread < 0)
    max_thread = nb_allowed_thread;

  if (TaskFactory::verboseLevel() >= 1)
    std::cout << "TBB: TBBTaskImplementation executeParallelFor begin=" << begin
              << " size=" << size << " max_thread=" << max_thread
              << " grain_size=" << options.grainSize()
              << " nb_allowed=" << nb_allowed_thread << '\n';

  // In sequential execution, call the method \a f directly.
  if (max_thread == 1 || max_thread == 0) {
    f->executeFunctor(begin, size);
    return;
  }

  // Replace the uninitialized values of \a options with those of \a m_default_loop_options
  ParallelLoopOptions true_options(options);
  true_options.mergeUnsetValues(TaskFactory::defaultParallelLoopOptions());
  true_options.setMaxThread(max_thread);

  ParallelForExecute pfe(this, true_options, begin, size, f, stat_info);

  tbb::task_arena* used_arena = nullptr;
  if (max_thread < nb_allowed_thread && max_thread < m_p->m_sub_arena_list.size())
    used_arena = m_p->m_sub_arena_list[max_thread];
  if (!used_arena)
    used_arena = &(m_p->m_main_arena);
  used_arena->execute(pfe);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
executeParallelFor(const ParallelFor1DLoopInfo& loop_info)
{
  _executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution of an N-dimensional loop.
 *
 * \warning The current implementation does not take into account \a options
 * for loops other than one dimension.
 */
template <int RankValue> void TBBTaskImplementation::
_executeMDParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                      IMDRangeFunctor<RankValue>* functor,
                      const ForLoopRunInfo& run_info)
{
  ParallelLoopOptions options;
  if (run_info.options().has_value())
    options = run_info.options().value();

  ScopedExecInfo sei(run_info);
  ForLoopOneExecStat* stat_info = sei.statInfo();
  ::Arcane::Impl::ScopedStatLoop scoped_loop(sei.isOwn() ? stat_info : nullptr);

  if (TaskFactory::verboseLevel() >= 1) {
    std::cout << "TBB: TBBTaskImplementation executeMDParallelFor nb_dim=" << RankValue
              << " nb_element=" << loop_ranges.nbElement()
              << " grain_size=" << options.grainSize()
              << " name=" << run_info.traceInfo().traceInfo()
              << " has_stat_info=" << (stat_info != nullptr)
              << '\n';
  }

  Integer max_thread = options.maxThread();
  // In sequential execution, call the method \a f directly.
  if (max_thread == 1 || max_thread == 0) {
    functor->executeFunctor(loop_ranges);
    return;
  }

  // Replace the uninitialized values of \a options with those of \a m_default_loop_options
  ParallelLoopOptions true_options(options);
  true_options.mergeUnsetValues(TaskFactory::defaultParallelLoopOptions());

  Integer nb_allowed_thread = m_p->nbAllowedThread();
  if (max_thread < 0)
    max_thread = nb_allowed_thread;
  tbb::task_arena* used_arena = nullptr;
  if (max_thread < nb_allowed_thread)
    used_arena = m_p->m_sub_arena_list[max_thread];
  if (!used_arena)
    used_arena = &(m_p->m_main_arena);

  // For now for dimension 1, use the historical 'ParallelForExecute'
  if constexpr (RankValue == 1) {
    auto range_1d = _toTBBRange(loop_ranges);
    auto x1 = [&](Integer begin, Integer size) {
      functor->executeFunctor(makeLoopRanges(ForLoopRange(begin, size)));
      //functor->executeFunctor(ComplexForLoopRanges<1>(begin,size));
    };
    LambdaRangeFunctorT<decltype(x1)> functor_1d(x1);
    Integer begin1 = CheckedConvert::toInteger(range_1d.dim(0).begin());
    Integer size1 = CheckedConvert::toInteger(range_1d.dim(0).size());
    ParallelForExecute pfe(this, true_options, begin1, size1, &functor_1d, stat_info);
    used_arena->execute(pfe);
  }
  else {
    MDParallelForExecute<RankValue> pfe(this, true_options, loop_ranges, functor, stat_info);
    used_arena->execute(pfe);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
executeParallelFor(Integer begin, Integer size, Integer grain_size, IRangeFunctor* f)
{
  ParallelLoopOptions opts(TaskFactory::defaultParallelLoopOptions());
  opts.setGrainSize(grain_size);
  ForLoopRunInfo run_info(opts);
  executeParallelFor(ParallelFor1DLoopInfo(begin, size, f, run_info));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
executeParallelFor(Integer begin, Integer size, const ParallelLoopOptions& options, IRangeFunctor* f)
{
  executeParallelFor(ParallelFor1DLoopInfo(begin, size, f, ForLoopRunInfo(options)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TBBTaskImplementation::TaskThreadInfo* TBBTaskImplementation::
currentTaskThreadInfo() const
{
  Int32 thread_id = currentTaskThreadIndex();
  if (thread_id >= 0)
    return m_p->threadTaskInfo(thread_id);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 TBBTaskImplementation::
currentTaskIndex() const
{
  Int32 thread_id = currentTaskThreadIndex();
  // This test was added to bypass a bug in one of the versions
  // of OneTBB. It is probably useless today (2025)
  if (thread_id < 0 || thread_id >= m_p->nbAllowedThread())
    return 0;
  TBBTaskImplementation::TaskThreadInfo* tti = currentTaskThreadInfo();
  if (tti) {
    Int32 task_index = tti->taskIndex();
    if (task_index >= 0)
      return task_index;
  }
  return thread_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTask::
launchAndWait()
{
  tbb::task_group task_group;
  task_group.run(taskFunctor());
  task_group.wait();
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTask::
launchAndWait(ConstArrayView<ITask*> tasks)
{
  tbb::task_group task_group;
  Integer n = tasks.size();
  if (n == 0)
    return;

  //set_ref_count(n+1);
  for (Integer i = 0; i < n; ++i) {
    auto* t = static_cast<OneTBBTask*>(tasks[i]);
    task_group.run(t->taskFunctor());
  }
  task_group.wait();
  for (Integer i = 0; i < n; ++i) {
    auto* t = static_cast<OneTBBTask*>(tasks[i]);
    delete t;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITask* OneTBBTask::
_createChildTask(ITaskFunctor* functor)
{
  auto* t = new OneTBBTask(functor);
  return t;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DI_REGISTER_PROVIDER(TBBTaskImplementation,
                            DependencyInjection::ProviderProperty("TBBTaskImplementation"),
                            ARCANE_DI_INTERFACES(ITaskImplementation),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
