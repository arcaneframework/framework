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

#include <alien/utils/Precomp.h>

#include <arccore/base/String.h>
#include <arccore/trace/ITraceMng.h>

#include <alien/expression/solver/ILinearSolver.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT SolverStat
{
 public:
  template <typename SolverT> friend class SolverStater;
  /** Constructeur de la classe */
  SolverStat();

  /** Destructeur de la classe */
  virtual ~SolverStat() {}

 public:
  Integer solveCount() const;
  Integer iterationCount() const;
  Real initializationTime() const;
  Real initializationCpuTime() const;
  Real prepareTime() const;
  Real prepareCpuTime() const;
  Real solveTime() const;
  Real solveCpuTime() const;

  Integer lastIterationCount() const;
  Real lastPrepareTime() const;
  Real lastPrepareCpuTime() const;
  Real lastSolveTime() const;
  Real lastSolveCpuTime() const;

  void reset();

 public:
  void print(
  ITraceMng* traceMng, const SolverStatus& status, String title = String()) const;

 protected:
  Integer m_solve_count;
  Integer m_iteration_count;
  Integer m_last_iteration_count;
  Real m_initialization_time, m_initialization_cpu_time;
  Real m_prepare_time, m_prepare_cpu_time;
  Real m_last_prepare_time, m_last_prepare_cpu_time;
  Real m_solve_time, m_solve_cpu_time;
  Real m_last_solve_time, m_last_solve_cpu_time;

 private:
  class InternalTraceSizer;
  void _internalPrint(std::ostream& o, const Integer prefix_size,
                      const SolverStatus& status, String title) const;
};

/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
