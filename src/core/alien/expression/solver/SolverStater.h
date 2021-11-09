/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT SolverStater : public SolverStat
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
  /** Constructeur de la classe */
  SolverStater();

  /** Destructeur de la classe */
  virtual ~SolverStater() {}

 public:
  void reset();

  void startInitializationMeasure();

  void stopInitializationMeasure();

  void startPrepareMeasure();

  void suspendPrepareMeasure(); //!< Incremental contribution for prepare phase.
  void stopPrepareMeasure();

  void startSolveMeasure();

  void stopSolveMeasure(const Alien::SolverStatus& status);

  static Real getVirtualTimeCounter() { return _getVirtualTime(); }

  static Real getRealTimeCounter() { return _getRealTime(); }

  class Sentry
  {
   public:
    Sentry(Real& time_counter, bool is_virtual = false)
    : m_counter(time_counter)
    , m_is_virtual(is_virtual)
    {
      m_start_counter = m_is_virtual ? SolverStater::getVirtualTimeCounter()
                                     : SolverStater::getRealTimeCounter();
    }
    virtual ~Sentry()
    {
      Real end_counter = m_is_virtual ? SolverStater::getVirtualTimeCounter()
                                      : SolverStater::getRealTimeCounter();
      m_counter += end_counter - m_start_counter;
    }

   private:
    Real& m_counter;
    Real m_start_counter;
    bool m_is_virtual;
  };

 private:
  static Arccore::Real _getVirtualTime();

  static Arccore::Real _getRealTime();

  static void _errorInTimer(const String& msg, int retcode);

  void _startTimer();

  void _stopTimer();

 private:
  eStateType m_state;
  Integer m_suspend_count;
  Real m_real_time; //!< 'wall clock' time for the lastest start or stop
  Real m_cpu_time; //!< 'cpu' time for the lastest start or stop
};

/*---------------------------------------------------------------------------*/

template <typename SolverT>
class SolverStatSentry
{
 private:
  bool m_is_released = false;
  Alien::SolverStatus& m_solver_status;
  SolverStater& m_solver_stater;
  SolverStater::eStateType m_state = SolverStater::eNone;

 public:
  SolverStatSentry(SolverT* solver, SolverStater::eStateType state)
  : m_solver_status(solver->getStatusRef())
  , m_solver_stater(solver->getSolverStater())
  , m_state(state)
  {
    switch (m_state) {
    case SolverStater::eInit:
      m_solver_stater.reset();
      m_solver_stater.startInitializationMeasure();
      break;
    case SolverStater::ePrepare:
      m_solver_stater.startPrepareMeasure();
      break;
    case SolverStater::eSolve:
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
    case SolverStater::eInit:
      m_solver_stater.stopInitializationMeasure();
      break;
    case SolverStater::ePrepare:
      m_solver_stater.stopPrepareMeasure();
      break;
    case SolverStater::eSolve:
      m_solver_stater.stopSolveMeasure(m_solver_status);
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
