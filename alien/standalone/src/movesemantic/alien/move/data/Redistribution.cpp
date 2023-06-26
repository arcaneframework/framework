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

#include "Redistribution.h"

namespace Alien::Move
{
MatrixData redistribute_matrix(Redistributor& redis, MatrixData&& src)
{
  auto multi = redis.redistribute(src.impl());
  auto forgotten = std::move(src); // Take ownership of src
  return createMatrixData(multi);
}

VectorData redistribute_vector(Redistributor& redis, VectorData&& src)
{
  auto multi = redis.redistribute(src.impl());
  auto forgotten = std::move(src); // Take ownership of src
  return createVectorData(multi);
}

VectorData redistribute_back_vector(Redistributor& redis, VectorData&& src)
{
  return std::move(src);
}
} // namespace Alien::Move