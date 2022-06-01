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

#include <alien/ref/AlienRefSemantic.h>
#include <alien/ref/handlers/scalar/VectorWriter.h>

#include <alien/trilinos/backend.h>
#include <alien/trilinos/options.h>

class SimpleLinearProblemFixtureRef : public ::testing::Test
{
 public:
  SimpleLinearProblemFixtureRef()
  {
    m_pm.reset(Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
    MPI_COMM_WORLD));

    auto size = 100;

    m_matrix = Alien::Matrix(size, size, m_pm.get());

    m_rhs = Alien::ones(size, m_pm.get());
  }

  void SetUp() override
  {
    const auto& dist = m_matrix.distribution();
    int offset = dist.rowOffset();
    int lsize = dist.localRowSize();
    int gsize = dist.globalRowSize();

    {
      Alien::DirectMatrixBuilder builder(
      m_matrix, Alien::DirectMatrixOptions::eResetValues);
      builder.reserve(3); // Réservation de 3 coefficients par ligne
      builder.allocate(); // Allocation de l'espace mémoire réservé

      for (int irow = offset; irow < offset + lsize; ++irow) {
        builder(irow, irow) = 2.;
        if (irow - 1 >= 0)
          builder(irow, irow - 1) = -1.;
        if (irow + 1 < gsize)
          builder(irow, irow + 1) = -1.;
      }
    }
  }

 public:
  Alien::Matrix m_matrix;
  Alien::Vector m_rhs;

 private:
  std::unique_ptr<Arccore::MessagePassing::IMessagePassingMng> m_pm;
};

TEST_F(SimpleLinearProblemFixtureRef, SimpleSolve)
{
  Alien::Vector x(m_matrix.distribution().rowDistribution());

  auto solver = Alien::Trilinos::LinearSolver();

  ASSERT_TRUE(solver.solve(m_matrix, m_rhs, x));
}

TEST_F(SimpleLinearProblemFixtureRef, ParametrizedSolve)
{
  Alien::Vector x(m_matrix.distribution().rowDistribution());

  auto options = Alien::Trilinos::Options()
                 .numIterationsMax(50)
                 .stopCriteriaValue(1e-10)
                 .preconditioner(Alien::Trilinos::OptionTypes::Relaxation)
                 .solver(Alien::Trilinos::OptionTypes::GMRES);

  auto solver = Alien::Trilinos::LinearSolver(options);

  ASSERT_TRUE(solver.solve(m_matrix, m_rhs, x));
}

TEST_F(SimpleLinearProblemFixtureRef, MultipleSolve)
{
  Alien::Vector x(m_matrix.distribution().rowDistribution());
  {
    Alien::LocalVectorWriter w(x);
    for (int i = 0; i < w.size(); i++) {
      w[i] = 1.0;
    }
    w.end();
  }
  {
    // First call, should fail
    auto options = Alien::Trilinos::Options()
                   .numIterationsMax(1)
                   .stopCriteriaValue(1e-20)
                   .preconditioner(Alien::Trilinos::OptionTypes::NoPC)
                   .solver(Alien::Trilinos::OptionTypes::CG);

    auto solver = Alien::Trilinos::LinearSolver(options);
    EXPECT_FALSE(solver.solve(m_matrix, m_rhs, x));
  }

  {
    Alien::LocalVectorWriter w(x);
    for (int i = 0; i < w.size(); i++) {
      w[i] = 1.0;
    }
    w.end();
  }

  {
    // Second call, should succeed
    auto solver = Alien::Trilinos::LinearSolver();
    ASSERT_TRUE(solver.solve(m_matrix, m_rhs, x));
  }
}
