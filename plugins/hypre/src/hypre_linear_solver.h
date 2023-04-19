/*
 * Copyright 2021 IFPEN-CEA
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

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

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
