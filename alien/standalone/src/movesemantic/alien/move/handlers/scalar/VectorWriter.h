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

#include <alien/handlers/scalar/BaseVectorWriter.h>

// FIXME: check if correct objects are used.

#include <alien/move/data/VectorData.h>
#include <alien/utils/MoveObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorWriter
: protected MoveObject<VectorData>
, public Common::VectorWriterT<Arccore::Real, Parameters<GlobalIndexer>>
{
 public:
  VectorWriter(VectorData&& vector)
  : MoveObject<VectorData>(std::move(vector))
  , Common::VectorWriterT<Arccore::Real, Parameters<GlobalIndexer>>(reference())
  {}

  VectorData&& release()
  {
    end();

    return MoveObject<VectorData>::release();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LocalVectorWriter
: protected MoveObject<VectorData>
, public Common::VectorWriterT<Arccore::Real, Parameters<LocalIndexer>>
{
 public:
  LocalVectorWriter(VectorData&& vector)
  : MoveObject<VectorData>(std::move(vector))
  , Common::VectorWriterT<Arccore::Real, Parameters<LocalIndexer>>(reference())
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
