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

#ifndef ALIEN_VECTORANALYTICS_H
#define ALIEN_VECTORANALYTICS_H

#include <alien/benchmark/export.h>

#include <alien/move/data/VectorData.h>

namespace Alien::Benchmark
{
struct VectorAnalytics
{
 public:
  Arccore::Real min;
  Arccore::Real max;
  Arccore::Real abs_min;
  Arccore::Real abs_max;
};

ALIEN_BENCHMARK_EXPORT VectorAnalytics computeAnalytics(const Alien::Move::VectorData& v);

} // namespace Alien::Benchmark

#endif //ALIEN_VECTORANALYTICS_H
