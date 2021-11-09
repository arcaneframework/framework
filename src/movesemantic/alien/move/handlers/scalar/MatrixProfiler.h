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

#include <alien/handlers/profiler/BaseMatrixProfiler.h>

#include <alien/utils/MoveObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MatrixProfiler : protected MoveObject<MatrixData>
, public Common::MatrixProfilerT<Real>
{
 public:
  MatrixProfiler(MatrixData&& matrix)
  : MoveObject<MatrixData>(std::move(matrix))
  , Common::MatrixProfilerT<Arccore::Real>(reference())
  {}

  MatrixData&& release()
  {
    allocate();

    return MoveObject<MatrixData>::release();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
