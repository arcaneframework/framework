// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Timer.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Management of a timer.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TIMER_H
#define ARCANE_CORE_TIMER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimerMng;
class ITimeStats;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of a timer.

 * An instance of this class allows measuring the time elapsed between its
 * activation via the start() method and its stopping via the stop() method.

 * The timer can be used multiple times, and it is possible to
 * know both the number of activations (nbActivated()) and the total time
 * spent in its successive activations (totalTime()).

 * There are two operating modes:
 * <ul>
 * <li>#TimerVirtual: the timer uses the CPU time of the process. This time
 * is constant regardless of the machine load;</li>
 * <li>#TimerReal: the timer uses real time. The resolution of this timer is
 * generally better than with the previous type, but it is only significant
 * when the machine is dedicated to the process.</li>
 * </ul>
 *
 * \note Since version 3.6 of %Arcane, #TimerVirtual is obsolete and
 * the returned value will be equivalent to #TimerReal.
 *
* The timer resolution depends on the machine. It is on the order of
* milliseconds for timers using CPU time and on the order of
* microseconds for timers using real time.
 */
class ARCANE_CORE_EXPORT Timer
{
 public:

  //! Timer type
  enum eTimerType
  {
    /*!
     * \brief Timer using CPU time (obsolete).
     *
     * \deprecated This timer is no longer used and behaves like
     * the clock time (TimerReal).
     */
    TimerVirtual,
    //! Timer using real time
    TimerReal
  };

 public:
  
  /*!
   * \brief Sentinel for the timer.
   * The sentinel associated with a timer allows it to be triggered
   * upon its construction and stopped upon its
   * destruction. This ensures that the timer will be properly stopped in case
   * of an exception, for example.
   */
  class ARCANE_CORE_EXPORT Sentry
  {
   public:
    //! Associates the timer \a t and starts it
    Sentry(Timer* t) : m_timer(t)
      { m_timer->start(); }
    //! Stops the associated timer
    ~Sentry()
      { m_timer->stop(); }
   private:
    Timer* m_timer; //!< Associated timer
  };

  /*!
   * \brief Positions the name of the currently executing action.
   *
   * An action name can be anything. It is
   * just used to differentiate the different parts of a
   * run and to know the time of each one.
   * Actions must be nested within each other
   */
  class ARCANE_CORE_EXPORT Action
  {
   public:
    Action(ISubDomain* sub_domain,const String& action_name,bool print_time=false);
    Action(ITimeStats* stats,const String& action_name,bool print_time=false);
    ~Action();
   public:
   private:
    ITimeStats* m_stats;
    String m_action_name;
    bool m_print_time;
   private:
    void _init();
  };

  /*!
   * \brief Positions the phase of the currently executing action.
   */
  class ARCANE_CORE_EXPORT Phase
  {
   public:
   public:
    Phase(ISubDomain* sub_domain,eTimePhase pt);
    Phase(ITimeStats* stats,eTimePhase pt);
    ~Phase();
   public:
   private:
    ITimeStats* m_stats; //!< Sub-domain manager
    eTimePhase m_phase_type;
   private:
    void _init();
  };

  /*!
   * \brief Displays the time elapsed between the call to the constructor and the destructor.
   *
   * This class allows simply displaying, at the time of the destructor,
   * the real time elapsed since the call to the constructor. The display is done
   * via the info() method of ITraceMng.
   * \code
   * {
   *   Timer::SimplePrinter sp(traceMng(),"myFunction");
   *   myFunction();
   * }
   * \endcode
   */
  class ARCANE_CORE_EXPORT SimplePrinter
  {
   public:
    SimplePrinter(ITraceMng* tm,const String& msg);
    SimplePrinter(ITraceMng* tm,const String& msg,bool is_active);
    ~SimplePrinter();
   private:
    ITraceMng* m_trace_mng;
    Real m_begin_time;
    bool m_is_active;
    String m_message;
   private:
    void _init();
  };

 public:

  /*!
   * \brief Constructs a timer.
   *
   * Constructs a timer linked to the sub-domain \a sd, with name \a name and
   * type \a type.
   */
  Timer(ISubDomain* sd,const String& name,eTimerType type);

  /*!
   * \brief Constructs a timer.
   *
   * Constructs a timer linked to the manager \a tm, with name \a name and
   * type \a type.
   */
  Timer(ITimerMng* tm,const String& name,eTimerType type);

  ~Timer(); //!< Frees resources

 public:
	
  /*!
   * \brief Activates the timer.
   *
   * If the timer is already active, this method does nothing.
   */
  void start();

  /*!
   * \brief Deactivates the timer.
   *
   * If the timer is not active at the time of the call, this method does not
   * do anything.
   *
   * \return the time elapsed (in seconds) since the last activation.
   */
  Real stop();

  //! Returns the activation status of the timer
  bool isActivated() const { return m_is_activated; }

  //! Returns the name of the timer
  const String& name() const { return m_name; }

  //! Returns the total time (in seconds) spent in the timer
  Real totalTime() const { return m_total_time; }

  //! Returns the time (in seconds) spent during the last activation of the timer
  Real lastActivationTime() const { return m_activation_time; }

  //! Returns the number of times the timer has been activated
  Integer nbActivated() const { return m_nb_activated; }

  //! Returns the type of time used
  eTimerType type() const { return m_type; }

  //! Resets the time counters
  void reset();

  //! Manager associated with this timer.
  ITimerMng* timerMng() const { return m_timer_mng; }
 public:
  static TimeMetricAction phaseAction(ITimeStats* s,eTimePhase phase);
 public:
  //! \internal
  void _setStartTime(Real t) { m_start_time = t; }
  //! \internal
  Real _startTime() const { return m_start_time; }
 private:

  ITimerMng* m_timer_mng; //!< Timer manager
  eTimerType m_type; //!< Timer type
  Integer m_nb_activated; //!< Number of times the timer has been activated
  bool m_is_activated; //!< \a true if the timer is active
  Real m_activation_time; //!< Time spent during the last activation
  Real m_total_time; //!< Total time spent in the timer
  String m_name; //!< Timer name
  Real m_start_time; //!< Time of the start of the last activation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
