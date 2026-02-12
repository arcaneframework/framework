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
  SolverStat() = default;

  /** Destructeur de la classe */
  virtual ~SolverStat() = default;

  Integer solveCount() const;
  Integer iterationCount() const;
  Real initializationTime() const;
  Real prepareTime() const;
  Real solveTime() const;

  Integer lastIterationCount() const;
  Real lastPrepareTime() const;
  Real lastSolveTime() const;

  void reset();

  void print(ITraceMng* traceMng, const SolverStatus& status, String title = String()) const;

  void finalPrint(ITraceMng* traceMng, String title = String()) const;

 protected:
  Integer m_solve_count = 0;
  Integer m_iteration_count = 0;
  Integer m_failed_solve_count = 0;
  Integer m_failed_iteration_count = 0;

  Real m_initialization_time = 0;
  Real m_prepare_time = 0;
  Real m_solve_time = 0;

  Real m_failed_prepare_time = 0;
  Real m_failed_solve_time = 0;

  Integer m_last_iteration_count;
  Real m_last_prepare_time = 0;
  Real m_last_solve_time = 0;

 private:
  class InternalTraceSizer;
  void _internalPrint(std::ostream& o, const Integer prefix_size,
                      const SolverStatus& status, String title) const;
  void _internalFinalPrint(std::ostream& o, String title) const;
};

/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
