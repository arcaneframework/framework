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

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/LinearSolverT.h>

#include <alien/ginkgo/backend.h>
#include <alien/ginkgo/options.h>
#include <alien/ginkgo/export.h>

#include "matrix.h"
#include "vector.h"

// ginkgo
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/log/convergence.hpp>

#include <memory>

namespace Alien
{
// Compile GinkgoLinearSolver.
template class ALIEN_GINKGO_EXPORT LinearSolver<BackEnd::tag::ginkgo>;
} // namespace Alien

namespace Alien::Ginkgo
{
class InternalLinearSolver
: public IInternalLinearSolver<Matrix, Vector>
, public ObjectWithTrace
{
 public:
  typedef SolverStatus Status;

  InternalLinearSolver() = default;

  explicit InternalLinearSolver(const Options& options)
  : m_status()
  , m_init_time(0.0)
  , m_total_solve_time(0.0)
  , m_solve_num(0)
  , m_total_iter_num(0)
  , m_stat()
  , m_options(options)
  {}

  ~InternalLinearSolver() override = default;

 public:
  // Nothing to do
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm) override {}

  bool solve(const Matrix& A, const Vector& b, Vector& x) override;

  bool hasParallelSupport() const override { return true; }

  //! Etat du solveur
  const SolverStatus& getStatus() const override;
  const SolverStat& getSolverStat() const override { return m_stat; }
  SolverStat& getSolverStat() override { return m_stat; }

  std::shared_ptr<ILinearAlgebra> algebra() const override;

 private:
  Status m_status; // to be uodated - DONE
  Arccore::Real m_init_time; // DONE
  Arccore::Real m_total_solve_time;
  Arccore::Integer m_solve_num; // DONE
  Arccore::Integer m_total_iter_num; // DONE
  SolverStat m_stat; // DONE
  Options m_options; // DONE

 private:
  using stop_iter_type = std::unique_ptr<gko::stop::Iteration::Factory, std::default_delete<gko::stop::Iteration::Factory>>;
  //using stop_res_type = std::unique_ptr<gko::stop::AbsoluteResidualNorm<>::Factory, std::default_delete<gko::stop::AbsoluteResidualNorm<>::Factory>>;
  using stop_res_type = std::unique_ptr<gko::stop::RelativeResidualNorm<>::Factory, std::default_delete<gko::stop::RelativeResidualNorm<>::Factory>>;
  using exec_type = std::shared_ptr<const gko::Executor>;

  void display_solver_infos(const Alien::Ginkgo::OptionTypes::eSolver& solver, const Alien::Ginkgo::OptionTypes::ePreconditioner& prec);
};
