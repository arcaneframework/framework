// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceMessage.h                                              (C) 2008      */
/*                                                                           */
/* Timer pour message de trace.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_TRACETIMER_H
#define ARCANE_UTILS_TRACETIMER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define DAY_TRACE_TIMER

#include <cassert>

#if   defined(MPI_TRACE_TIMER)
#elif defined(DAY_TRACE_TIMER)
#elif defined(CPU_TRACE_TIMER)
#elif defined(SCPU_TRACE_TIMER)
#else
#error "TraceTimer type not defined"
/* MPI_TRACE_TIMER  : utilise la fonction MPI_Wtime de MPI 
 * DAY_TRACE_TIMER  : mesure le temps qui passe (** SVr4, BSD 4.3 **)
 * CPU_TRACE_TIMER  : mesure le nombre de 'ticks' CPU (** SVR4, SVID, POSIX, X/OPEN, BSD 4.3 **)
 *                  : (on peut distinguer le processus même de ses enfants et appels système)
 *                  : pour plus d'infos voir 'man times'
 * SCPU_TRACE_TIMER : comme CPU_TRACE_TIMER mais inclus les temps système et des enfants
 *
 * On peut aussi utiliser 
 * return ((double) clock())/CLOCKS_PER_SEC;
 * qui compte les cycles consommés et limite à 72min (arch 32bits) [need <time.h>]
 * ou getrusage qui ressemble à times() mais fourni plus de paramètres
 */
#endif

#if defined(MPI_TRACE_TIMER)
#include "mpi.h"
class InnerTimer_MPI {
protected:
  double systemTime() const {
    return MPI_Wtime();
  }
public:
  static const char * type() { return "MPI Timer"; }
};
#endif /* MPI_TRACE_TIMER */

#if defined(DAY_TRACE_TIMER)
class InnerTimer_DAY {
protected:
  double systemTime() {
    return platform::getRealTime();
  }
public:
  static const char * type() { return "Day Timer"; }
};
#endif /* DAY_TRACE_TIMER */

#if defined(CPU_TRACE_TIMER)
#include <unistd.h>    // sysconf()
#include <sys/times.h> // times()
class InnerTimer_CPU {
protected:
  struct tms tp;
  double systemTime() {
    static const long _CLK_TCK = sysconf(_SC_CLK_TCK);
    times(&tp);
    return static_cast<double>(tp.tms_utime)/_CLK_TCK; // user only
  }
public:
  static const char * type() { return "Cpu Timer"; }
};
#endif /* CPU_TRACE_TIMER */

#if defined(SCPU_TRACE_TIMER)
#include <unistd.h>    // sysconf()
#include <sys/times.h> // times()
class InnerTimer_SysCPU {
protected:
  struct tms tp;
  double systemTime() {
    static const long _CLK_TCK = sysconf(_SC_CLK_TCK);
    return static_cast<double>(times(&tp))/_CLK_TCK; // user+system+children
  }
public:
  static const char * type() { return "SysCpu Timer"; }
};
#endif /* SCPU_TRACE_TIMER */


template<typename Model>
class TraceTimerT : public Model {
public:
  enum ClockState { init, stopped, running };
private:
  //! Timer State
  ClockState state;

  //! Initial and last time
  double t0,t1;

  //! Cumulate time
  double total;
public:
  //! New timer
  /*! Autostart by default */
  TraceTimerT(const bool _start = true)
    : state(init), t0(0), t1(0), total(0) { 
    if (_start) start();
  }

  //! reset timer
  void reset() {
    state = init;
    total = 0;
    t0 = t1 = 0;
  }

  //! start the timer or restart without cumulate
  void start() {
    state = running;
    t0 = this->systemTime();
  }

  //! start or restart the timer and cumuluate
  /*! Usefull for starting the count of a new partial time and keep the cumulative timer */
  void restart() {
    if (state == running) {
      t1 = this->systemTime();
      total += t1 - t0;
    }
    state = running;
    t0 = this->systemTime();
  }

  //! stop timer
  double stop() {
    assert(state == running);
    state = stopped;
    t1 = this->systemTime();
    total += t1 - t0;
    return t1 - t0;
  }
  
  //! return state of timer
  ClockState getState() const {
    return state;
  }

  //! get partial time
  double getTime() {
    assert(state != init);
    if (state == running)
      t1 = this->systemTime();
    return t1 - t0;
  }

  //! get total time
  double getCumulTime() {
    assert(state != init);
    if (state == running)
      return total + this->systemTime() - t0;
    // si state stopped
    assert(state == stopped);
    return total;
  }
};

#if   defined(MPI_TRACE_TIMER)
typedef TraceTimerT<InnerTimer_MPI>     TraceTimer;
#elif defined(DAY_TRACE_TIMER)
typedef TraceTimerT<InnerTimer_DAY>     TraceTimer;
#elif defined(CPU_TRACE_TIMER)
typedef TraceTimerT<InnerTimer_CPU>     TraceTimer;
#elif defined(SCPU_TRACE_TIMER)
typedef TraceTimerT<InnerTimer_SysCPU>  TraceTimer;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_UTILS_TRACETIMER_H */
