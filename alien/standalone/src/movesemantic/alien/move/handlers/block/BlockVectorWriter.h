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

// FIXME: check if correct object is used.
#include <alien/handlers/block/BaseBlockVectorWriter.h>
#include <alien/move/data/VectorData.h>
#include <alien/utils/MoveObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BlockVectorWriter : protected MoveObject<VectorData>
, public Common::BlockVectorWriterT<Real>
{
 public:
  BlockVectorWriter(VectorData&& vector)
  : MoveObject<VectorData>(std::move(vector))
  , Common::BlockVectorWriterT<Arccore::Real>(reference())
  {}

  VectorData&& release()
  {
    end();

    return MoveObject<VectorData>::release();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LocalBlockVectorWriter : protected MoveObject<VectorData>
, public Common::LocalBlockVectorWriterT<Real>
{
 public:
  LocalBlockVectorWriter(VectorData&& vector)
  : MoveObject<VectorData>(std::move(vector))
  , Common::LocalBlockVectorWriterT<Arccore::Real>(reference())
  {}

  VectorData&& release()
  {
    end();

    return MoveObject<VectorData>::release();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
