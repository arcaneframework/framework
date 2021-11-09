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
#include <iostream>
#include <utility>

#include <gtest/gtest.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/kernels/dok/DoKMatrixT.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/dok/DoKBackEnd.h>

#include <alien/move/handlers/scalar/DirectMatrixBuilder.h>
#include <alien/move/data/MatrixData.h>

#include <Environment.h>

TEST(TestDoKMatrix, MultiImplConverter)
{
  // Build a SimpleCSR matrix
  Alien::MatrixDistribution mdist(4, 4, AlienTest::Environment::parallelMng());
  Alien::Space row_space(4, "Space");
  Alien::Space col_space(4, "Space");
  Alien::Move::MatrixData A(row_space, col_space, mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  Alien::Move::DirectMatrixBuilder builder(std::move(A), tag);
  builder.reserve(5);
  builder.allocate();

  auto first = mdist.rowOffset();
  auto last = first + mdist.localRowSize();

  if (first <= 0 && 0 < last)
    builder(0, 0) = -1.;
  if (first <= 1 && 1 < last)
    builder(1, 1) = -2.;
  if (first <= 2 && 2 < last) {
    builder(2, 2) = -3.;
    builder(2, 3) = 3.14;
  }
  if (first <= 3 && 3 < last) {
    builder(3, 1) = 2.71;
    builder(3, 3) = -4;
  }
  builder.finalize();

  std::cerr << builder.stats() << std::endl;

  A = builder.release();

  Alien::MultiMatrixImpl* multiA = A.impl();
  const Alien::DoKMatrix& dok_a = multiA->get<Alien::BackEnd::tag::DoK>();
  dok_a.backend();
}
