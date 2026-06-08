// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Profiling.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Classes to manage profiling.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_PROFILING_H
#define ARCCORE_BASE_PROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <atomic>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{
class AcceleratorStatInfoList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing retrieval of the time spent between the constructor
 * call and the destructor call.
 */
class ARCCORE_BASE_EXPORT ScopedStatLoop
{
 public:

  explicit ScopedStatLoop(ForLoopOneExecStat* s);
  ~ScopedStatLoop();

 public:

  Int64 m_begin_time = 0.0;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Loop execution statistics.
 */
class ARCCORE_BASE_EXPORT ForLoopStatInfoList
{
 public:

  ForLoopStatInfoList();
  ~ForLoopStatInfoList();

 public:

  void merge(const ForLoopOneExecStat& loop_stat_info, const ForLoopTraceInfo& loop_trace_info);

 public:

  /*!
   * \internal
   * \brief Opaque type for internal implementation.
   */
  ForLoopStatInfoListImpl* _internalImpl() const { return m_p; }

 private:

  ForLoopStatInfoListImpl* m_p = nullptr;
};

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to manage the profiling of a single loop execution.
 */
class ARCCORE_BASE_EXPORT ForLoopOneExecStat
{
 public:

  /*!
   * \brief Increments the number of chunks used.
   *
   * This method can be called simultaneously by multiple threads.
   */
  void incrementNbChunk() { ++m_nb_chunk; }

  //! Sets the loop start time (in nanoseconds)
  void setBeginTime(Int64 v) { m_begin_time = v; }

  //! Sets the loop end time in nanoseconds
  void setEndTime(Int64 v) { m_end_time = v; }

  //! Number of chunks
  Int64 nbChunk() const { return m_nb_chunk; }

  /*!
   * \brief Execution time (in nanoseconds).
   *
   * The returned value is only valid if setBeginTime() and setEndTime()
   * were called previously.
   */
  Int64 execTime() const { return m_end_time - m_begin_time; }

  void reset()
  {
    m_nb_chunk = 0;
    m_begin_time = 0;
    m_end_time = 0;
  }

 private:

  //! Number of loop decomposition chunks (in multi-thread)
  std::atomic<Int64> m_nb_chunk = 0;

  // Execution start time
  Int64 m_begin_time = 0;

  // Execution end time
  Int64 m_end_time = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Profiling manager.
 *
 * It is possible to enable profiling by calling setProfilingLevel() with
 * a value greater than or equal to 1.
 *
 * Statistics are added by retrieving an instance of
 * impl::ForLoopStatInfoList specific to the currently executing thread.
 */
class ARCCORE_BASE_EXPORT ProfilingRegistry
{
 public:

  /*!
   * TODO: Deprecate. Use
   * static impl::ForLoopStatInfoList* _threadLocalForLoopInstance()
   * instead.
   */
  ARCCORE_DEPRECATED_REASON("Y2023: Use _threadLocalForLoopInstance() instead")
  static Impl::ForLoopStatInfoList* threadLocalInstance();

  /*!
   * \brief Sets the profiling level.
   *
   * If 0, there is no profiling. Profiling is active starting from level 1.
   */
  static void setProfilingLevel(Int32 level);

  //! Profiling level
  static Int32 profilingLevel() { return m_profiling_level; }

  //! Indicates if profiling is active.
  static bool hasProfiling() { return m_profiling_level > 0; }

  /*!
   * \brief Visits the loop statistics list
   *
   * There is an instance of impl::ForLoopStatInfoList per thread that
   * executed a loop.
   *
   * This method must not be called if loops are currently executing.
   */
  static void visitLoopStat(const std::function<void(const Impl::ForLoopStatInfoList&)>& f);

  /*!
   * \brief Visits the accelerator statistics list
   *
   * There is an instance of impl::AcceleratorStatInfoList per thread
   * that executed a loop.
   *
   * This method must not be called when profiling is active.
   */
  static void visitAcceleratorStat(const std::function<void(const Impl::AcceleratorStatInfoList&)>& f);

  static const Impl::ForLoopCumulativeStat& globalLoopStat();

 public:

  // Public API but reserved for Arcane.

  /*!
   * \internal.
   * Thread-local instance of the loop statistics manager
   */
  static Impl::ForLoopStatInfoList* _threadLocalForLoopInstance();

  /*!
   * \internal.
   * Thread-local instance of the accelerator statistics manager
   */
  static Impl::AcceleratorStatInfoList* _threadLocalAcceleratorInstance();

 private:

  static Int32 m_profiling_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
