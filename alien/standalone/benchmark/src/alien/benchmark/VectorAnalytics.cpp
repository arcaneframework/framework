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

#include "VectorAnalytics.h"

#include <algorithm>

#include <arccore/message_passing/Messages.h>

#include <alien/move/handlers/scalar/VectorReader.h>

namespace Alien::Benchmark
{

VectorAnalytics computeAnalytics(const Alien::Move::VectorData& v)
{
  VectorAnalytics out;

  auto pm = v.distribution().parallelMng();

  Alien::Move::LocalVectorReader reader(std::move(v));

  out.min = reader[0];
  out.max = reader[0];
  out.abs_min = std::abs(reader[0]);
  out.abs_max = std::abs(reader[0]);

  for (auto i = 1; i < reader.size(); i++) {
    out.min = std::min(out.min, reader[i]);
    out.max = std::max(out.max, reader[i]);
    out.abs_min = std::min(out.abs_min, std::abs(reader[i]));
    out.abs_max = std::min(out.abs_max, std::abs(reader[i]));
  }

  out.min = Arccore::MessagePassing::mpAllReduce(pm, Arccore::MessagePassing::ReduceMin, out.min);
  out.abs_min = Arccore::MessagePassing::mpAllReduce(pm, Arccore::MessagePassing::ReduceMin, out.abs_min);
  out.max = Arccore::MessagePassing::mpAllReduce(pm, Arccore::MessagePassing::ReduceMin, out.max);
  out.abs_max = Arccore::MessagePassing::mpAllReduce(pm, Arccore::MessagePassing::ReduceMin, out.abs_max);

  return out;
}
} // namespace Alien::Benchmark