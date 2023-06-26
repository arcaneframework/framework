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
// Created by chevalierc on 23/02/22.
//

#ifndef ALIEN_ILINEARPROBLEM_H
#define ALIEN_ILINEARPROBLEM_H

#include <alien/move/data/MatrixData.h>
#include <alien/move/data/VectorData.h>

#include <alien/benchmark/export.h>

namespace Alien::Benchmark
{

class ILinearProblem
{
 public:
  virtual ~ILinearProblem() = default;

  ALIEN_BENCHMARK_EXPORT virtual Alien::Move::MatrixData matrix() const = 0;

  ALIEN_BENCHMARK_EXPORT virtual Alien::Move::VectorData vector() const = 0;
};

ALIEN_BENCHMARK_EXPORT std::unique_ptr<ILinearProblem> buildFromMatrixMarket(Arccore::MessagePassing::IMessagePassingMng* pm, const std::string& matrix_name, const std::string& rhs_name = "");

} // namespace Alien::Benchmark

#endif //ALIEN_ILINEARPROBLEM_H
