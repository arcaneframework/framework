// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/core/backend/LinearSolverT.h>
#include <alien/expression/solver/SolverStater.h>

#include <alien/hypre/backend.h>
#include <alien/hypre/export.h>
#include <alien/hypre/options.h>

#include "hypre_matrix.h"
#include "hypre_vector.h"

namespace Alien::Hypre
{
class InternalLinearSolver : public IInternalLinearSolver<Matrix, Vector>
, public ObjectWithTrace
{
 public:
  using Status = SolverStatus;

  InternalLinearSolver() = default;

  explicit InternalLinearSolver(const Options& options)
  : m_status()
  , m_options(options)
  {}

  ~InternalLinearSolver() final = default;

  void updateParallelMng(ALIEN_UNUSED_PARAM Arccore::MessagePassing::IMessagePassingMng* pm) final
  {
    // Nothing to do
  }

  bool solve(const Matrix& A, const Vector& b, Vector& x) override;

  bool hasParallelSupport() const final { return true; }

  //! Etat du solveur
  const Status& getStatus() const final;

  const SolverStat& getSolverStat() const final { return m_stat; }

  SolverStat& getSolverStat() override { return m_stat; }

  std::shared_ptr<ILinearAlgebra> algebra() const final;

 private:
  Status m_status;

  Arccore::Real m_init_time = 0.0;
  Arccore::Real m_total_solve_time = 0.0;
  Arccore::Integer m_solve_num = 0;
  Arccore::Integer m_total_iter_num = 0;

  SolverStat m_stat;
  Options m_options;

  void checkError(const Arccore::String& msg, int ierr, int skipError = 0) const;
};
} // namespace Alien::Hypre
