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

#include <gtest/gtest.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>

#include <alien/move/AlienMoveSemantic.h>
#include <alien/move/handlers/scalar/VectorWriter.h>

#include <alien/hypre/backend.h>
#include <alien/hypre/options.h>

class SimpleLinearProblemFixtureMove : public ::testing::Test
{
 public:
  SimpleLinearProblemFixtureMove()
  {
    m_pm.reset(Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
    MPI_COMM_WORLD));
    auto size = 100;

    m_distribution = Alien::MatrixDistribution(size, size, m_pm.get());

    m_matrix = Alien::Move::MatrixData(m_distribution);

    m_rhs = Alien::Move::VectorData(m_distribution.rowDistribution());
  }

  void SetUp() override
  {
    const auto& dist = m_matrix.distribution();
    int offset = dist.rowOffset();
    int lsize = dist.localRowSize();
    int gsize = dist.globalRowSize();

    {
      Alien::Move::DirectMatrixBuilder builder(
      std::move(m_matrix), Alien::DirectMatrixOptions::eResetAllocation);
      builder.reserve(3); // Réservation de 3 coefficients par ligne
      builder.allocate(); // Allocation de l'espace mémoire réservé

      for (int irow = offset; irow < offset + lsize; ++irow) {
        builder(irow, irow) = 2.;
        if (irow - 1 >= 0)
          builder(irow, irow - 1) = -1.;
        if (irow + 1 < gsize)
          builder(irow, irow + 1) = -1.;
      }

      m_matrix = builder.release();
    }

    {
      Alien::Move::LocalVectorWriter v_build(std::move(m_rhs));
      for (int i = 0; i < v_build.size(); i++) {
        v_build[i] = 1.0;
      }
      m_rhs = v_build.release();
    }
  }

 public:
  Alien::MatrixDistribution m_distribution;
  Alien::Move::MatrixData m_matrix;
  Alien::Move::VectorData m_rhs;

 private:
  std::unique_ptr<Arccore::MessagePassing::IMessagePassingMng> m_pm;
};

TEST_F(SimpleLinearProblemFixtureMove, SimpleSolve)
{
  Alien::Move::VectorData x(m_matrix.distribution().rowDistribution());

  auto solver = Alien::Hypre::LinearSolver();

  ASSERT_TRUE(solver.solve(m_matrix, m_rhs, x));
}

TEST_F(SimpleLinearProblemFixtureMove, ParametrizedSolve)
{
  Alien::Move::VectorData x(m_matrix.distribution().rowDistribution());

  auto options = Alien::Hypre::Options()
                 .numIterationsMax(12)
                 .stopCriteriaValue(1e-10)
                 .preconditioner(Alien::Hypre::OptionTypes::AMGPC)
                 .solver(Alien::Hypre::OptionTypes::GMRES);

  auto solver = Alien::Hypre::LinearSolver(options);

  ASSERT_TRUE(solver.solve(m_matrix, m_rhs, x));
}

TEST_F(SimpleLinearProblemFixtureMove, MultipleSolve)
{
  Alien::Move::VectorData x(m_matrix.distribution().rowDistribution());

  {
    // First call, should fail
    Alien::Move::LocalVectorWriter w(std::move(x));
    for (int i = 0; i < w.size(); i++) {
      w[i] = 1.0;
    }
    x = w.release();

    auto options = Alien::Hypre::Options()
                   .numIterationsMax(1)
                   .stopCriteriaValue(1e-20)
                   .preconditioner(Alien::Hypre::OptionTypes::NoPC)
                   .solver(Alien::Hypre::OptionTypes::CG);
    auto solver = Alien::Hypre::LinearSolver(options);
    EXPECT_FALSE(solver.solve(m_matrix, m_rhs, x));
    std::cerr << "Residual " << solver.getStatus().residual << " after "
              << solver.getStatus().iteration_count << std::endl;
  }

  {
    // Second call, should succeed
    auto solver = Alien::Hypre::LinearSolver();
    ASSERT_TRUE(solver.solve(m_matrix, m_rhs, x));
  }
}
