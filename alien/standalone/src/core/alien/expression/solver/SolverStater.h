// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* alienc                                         (C) 2000-2024              */
/*                                                                           */
/* Stats for Alien solvers                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#pragma once

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
class ALIEN_EXPORT BaseSolverStater
{
 public:
  typedef enum
  {
    eNone,
    eInit,
    ePrepare,
    eSolve
  } eStateType;

 public:
  BaseSolverStater()
  : m_state(eNone)
  , m_suspend_count(0)
  {}

  virtual ~BaseSolverStater() {}

 public:
  static Real getVirtualTimeCounter() { return _getVirtualTime(); }

  static Real getRealTimeCounter() { return _getRealTime(); }

  class Sentry
  {
   public:
    Sentry(Real& time_counter, bool is_virtual = false)
    : m_counter(time_counter)
    , m_is_virtual(is_virtual)
    {
      m_start_counter = m_is_virtual ? BaseSolverStater::getVirtualTimeCounter()
                                     : BaseSolverStater::getRealTimeCounter();
    }

    virtual ~Sentry()
    {
      Real end_counter = m_is_virtual ? BaseSolverStater::getVirtualTimeCounter()
                                      : BaseSolverStater::getRealTimeCounter();
      m_counter += end_counter - m_start_counter;
    }

   private:
    Real& m_counter;
    Real m_start_counter;
    bool m_is_virtual;
  };

 protected:
  static Arccore::Real _getVirtualTime();

  static Arccore::Real _getRealTime();

  static void _errorInTimer(const String& msg, int retcode);

  void _startTimer();

  void _stopTimer();

 protected:
  eStateType m_state;
  Integer m_suspend_count;
  Real m_real_time; //!< 'wall clock' time for the lastest start or stop
  Real m_cpu_time; //!< 'cpu' time for the lastest start or stop
};

template <typename SolverT>
class SolverStater : public BaseSolverStater
{
 public:
 public:
  /** Constructeur de la classe */
  SolverStater(SolverT* solver)
  : BaseSolverStater()
  , m_solver(solver)
  {}

  /** Destructeur de la classe */
  virtual ~SolverStater() {}

 public:
  void reset()
  {
    m_solver->getSolverStat().reset();

    m_state = eNone;
    m_suspend_count = 0;
  }

  void startInitializationMeasure()
  {
    ALIEN_ASSERT((m_state == eNone), ("Unexpected SolverStater state %d", m_state));
    _startTimer();
    m_state = eInit;
  }

  void stopInitializationMeasure()
  {
    ALIEN_ASSERT((m_state == eInit), ("Unexpected SolverStater state %d", m_state));
    _stopTimer();
    m_state = eNone;

    auto& solver_stat = m_solver->getSolverStat();
    solver_stat.m_initialization_time += m_real_time;
  }

  void startPrepareMeasure()
  {
    ALIEN_ASSERT((m_state == eNone), ("Unexpected SolverStater state %d", m_state));
    _startTimer();
    m_state = ePrepare;
  }

  void suspendPrepareMeasure() //!< Incremental contribution for prepare phase.
  {
    ALIEN_ASSERT((m_state == ePrepare), ("Unexpected SolverStater state %d", m_state));
    _stopTimer();
    auto& solver_stat = m_solver->getSolverStat();
    if (m_suspend_count == 0) {
      solver_stat.m_last_prepare_time = m_real_time;
    }
    else {
      solver_stat.m_last_prepare_time += m_real_time;
    }
    m_state = eNone;
    ++m_suspend_count;
  }

  void stopPrepareMeasure()
  {
    if (m_state == ePrepare)
      suspendPrepareMeasure();
    ALIEN_ASSERT((m_suspend_count > 0), ("Unexpected suspend count"));

    auto& solver_stat = m_solver->getSolverStat();
    solver_stat.m_last_prepare_time += m_real_time;

    m_suspend_count = 0;
    m_state = eNone;
    solver_stat.m_prepare_time += solver_stat.m_last_prepare_time;
  }

  void startSolveMeasure()
  {
    ALIEN_ASSERT((m_state == eNone), ("Unexpected SolverStater state %d", m_state));
    _startTimer();
    m_state = eSolve;
  }

  void stopSolveMeasure()
  {
    ALIEN_ASSERT((m_state == eSolve), ("Unexpected SolverStater state %d", m_state));
    _stopTimer();
    m_state = eNone;
    auto const& status = m_solver->getStatus();
    auto& solver_stat = m_solver->getSolverStat();
    solver_stat.m_last_solve_time = m_real_time;
    solver_stat.m_solve_time += solver_stat.m_last_solve_time;
    ++solver_stat.m_solve_count;
    solver_stat.m_last_iteration_count = status.iteration_count;
    solver_stat.m_iteration_count += solver_stat.m_last_iteration_count;
  }

 private:
  SolverT* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
template <typename SolverT>
class SolverStatSentry
{
 private:
  bool m_is_released = false;
  SolverStater<SolverT> m_solver_stater;
  BaseSolverStater::eStateType m_state = BaseSolverStater::eNone;

 public:
  SolverStatSentry(SolverStater<SolverT>& parent, BaseSolverStater::eStateType state)
  : m_solver_stater(parent)
  , m_state(state)
  {
    switch (m_state) {
    case BaseSolverStater::eInit:
      m_solver_stater.reset();
      m_solver_stater.startInitializationMeasure();
      break;
    case BaseSolverStater::ePrepare:
      m_solver_stater.startPrepareMeasure();
      break;
    case BaseSolverStater::eSolve:
      m_solver_stater.startSolveMeasure();
      break;
    default:
      break;
    }
  }

  virtual ~SolverStatSentry() { release(); }

  void release()
  {
    if (m_is_released)
      return;
    switch (m_state) {
    case BaseSolverStater::eInit:
      m_solver_stater.stopInitializationMeasure();
      break;
    case BaseSolverStater::ePrepare:
      m_solver_stater.stopPrepareMeasure();
      break;
    case BaseSolverStater::eSolve:
      m_solver_stater.stopSolveMeasure();
      break;
    default:
      break;
    }
    m_is_released = true;
  }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
