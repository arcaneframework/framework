#pragma once
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

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
                      const SolverStatus& status, const String& title) const;
  void _internalFinalPrint(std::ostream& o, const String& title) const;
};

/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
