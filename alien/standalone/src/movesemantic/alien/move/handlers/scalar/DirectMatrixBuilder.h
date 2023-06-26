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

#include <alien/handlers/scalar/BaseDirectMatrixBuilder.h>
#include <alien/utils/MoveObject.h>

#include <alien/move/data/MatrixData.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectMatrixBuilder : protected MoveObject<MatrixData>
, public Common::DirectMatrixBuilder
{
 public:
  using Common::DirectMatrixBuilder::ReserveFlag;
  using Common::DirectMatrixBuilder::ResetFlag;
  using Common::DirectMatrixBuilder::SymmetricFlag;

  DirectMatrixBuilder(MatrixData&& matrix, const ResetFlag reset_flag,
                      const SymmetricFlag symmetric_flag = SymmetricFlag::eSymmetric)
  : MoveObject<MatrixData>(std::move(matrix))
  , Common::DirectMatrixBuilder(reference(), reset_flag, symmetric_flag)
  {}

  virtual ~DirectMatrixBuilder() = default;

  MatrixData&& release()
  {
    finalize();

    return MoveObject<MatrixData>::release();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
