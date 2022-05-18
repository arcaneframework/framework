/*
 * Copyright 2022 IFPEN-CEA
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
 *  SPDX-License-Identifier: Apache-2.0
 */

//
// Created by chevalierc on 22/02/2022.
//

#ifndef ALIEN_MATRIXMARKETPROBLEM_H
#define ALIEN_MATRIXMARKETPROBLEM_H

#include <string>

#include <arccore/message_passing/IMessagePassingMng.h>

#include <alien/benchmark/ILinearProblem.h>

namespace Alien::Benchmark
{

class MatrixMarketProblem : public ILinearProblem
{
 public:
  MatrixMarketProblem(Arccore::MessagePassing::IMessagePassingMng* pm, const std::string& matrix_filename, const std::string& rhs_filename);
  MatrixMarketProblem(Arccore::MessagePassing::IMessagePassingMng* pm, const std::string& matrix_filename);

  virtual ~MatrixMarketProblem() = default;

  Alien::Move::MatrixData matrix() const override;

  Alien::Move::VectorData vector() const override;

 private:
  Alien::Move::MatrixData m_matrix;
  Alien::Move::VectorData m_rhs;
};
} // namespace Alien::Benchmark
#endif //ALIEN_MATRIXMARKETPROBLEM_H
