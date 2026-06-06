// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeStats.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface managing execution time statistics.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMESTATS_H
#define ARCANE_CORE_ITIMESTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Properties;
class Timer;
class JSONWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface managing execution time statistics.
 *
 * You must call beginGatherStats() to start collecting the
 * information and call endGatherStats() to stop the collection.
 *
 * Generally, this interface is not used directly but through
 * the Timer::Phase and Timer::Action classes.
 *
 * The methods of this class must only be called by a single thread.
 */
class ITimeStats
{
 public:

  // Releases resources.
  virtual ~ITimeStats() = default;

 public:

  //! Starts time collection
  virtual void beginGatherStats() = 0;
  //! Stops time collection
  virtual void endGatherStats() = 0;

 public:

  virtual void beginAction(const String& action_name) = 0;
  virtual void endAction(const String& action_name, bool print_time) = 0;

  virtual void beginPhase(eTimePhase phase) = 0;
  virtual void endPhase(eTimePhase phase) = 0;

 public:

  /*!
   * \brief Real elapsed time for phase \a phase
   *
   * Returns the real elapsed time (in seconds) for phase \a phase.
   */
  virtual Real elapsedTime(eTimePhase phase) = 0;

  /*!
   * \brief Elapsed time for a phase of an action.
   *
   * Returns the real elapsed time (in seconds) for phase \a phase
   * of action \a action. The returned time includes that of the action and
   * all of its children.
   */
  virtual Real elapsedTime(eTimePhase phase, const String& action) = 0;

  /*!
   * \brief Displays statistics for an action.
   *
   * Displays the statistics for action \a name as well as its sub-actions
   * for the current iteration.
   */
  virtual void dumpCurrentStats(const String& name) = 0;

 public:

  /*!
   * \brief Displays execution time statistics.
   *
   * It is possible to specify a value to get a time
   * per iteration or per entity. If \a use_elapsed_time is true,
   * it uses clock time; otherwise, it uses CPU time.
   */
  virtual void dumpStats(std::ostream& ostr, bool is_verbose, Real nb,
                         const String& name, bool use_elapsed_time = false) = 0;

  /*!
   * \brief Displays the current date and memory consumption.
   *
   * This operation is collective on \a pm.
   *
   * This operation displays the memory consumed for the current subdomain
   * as well as the min and max for all subdomains.
   */
  virtual void dumpTimeAndMemoryUsage(IParallelMng* pm) = 0;

  /*!
   * \brief Indicates if statistics are active.
   *
   * Statistics are active between the call to beginGatherStats()
   * and endGatherStats().
   */
  virtual bool isGathering() const = 0;

  //! Serializes the temporal statistics into the writer \a writer.
  virtual void dumpStatsJSON(JSONWriter& writer) = 0;

  //! Associated collection interface
  virtual ITimeMetricCollector* metricCollector() = 0;

  /*!
   * \brief Notifies that a new iteration of the calculation loop begins.
   *
   * This information is used to calculate times per iteration.
   */
  virtual void notifyNewIterationLoop() = 0;
  virtual void saveTimeValues(Properties* p) = 0;
  virtual void mergeTimeValues(Properties* p) = 0;

  /*
   * \brief Resets the current statistics for an action and its sub-actions
   *
   * Resets the statistics for action \a action_name and its
   * sub-actions. If no action named \a action_name exists, it does nothing.
   *
   * This method is reserved for testing and should not be used
   * outside of this configuration to avoid invalidating the
   * temporal statistics.
   */
  virtual void resetStats(const String& action_name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
