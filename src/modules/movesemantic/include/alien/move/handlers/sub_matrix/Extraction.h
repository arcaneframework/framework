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

#include <alien/move/data/MatrixData.h>
#include <alien/utils/Precomp.h>

#include <alien/data/utils/ExtractionIndices.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Move
{

  class ALIEN_EXPORT SubMatrix
  {
   public:
    static MatrixData Extract(const IMatrix& matrix, const ExtractionIndices& indices);

   private:
    static MatrixData extractRange(const IMatrix& matrix, const ExtractionIndices& indices);

    static MatrixData extractIndices(
    const IMatrix& matrix, const ExtractionIndices& indices);
  };

} // namespace Move
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
