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
#include <alien/handlers/block/BaseBlockVectorReader.h>
#include <alien/move/data/VectorData.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BlockVectorReader
: public Common::BlockVectorReaderT<Arccore::Real, Parameters<GlobalIndexer>>
{
 public:
  BlockVectorReader(const VectorData& vector)
  : Common::BlockVectorReaderT<Arccore::Real, Parameters<GlobalIndexer>>(vector)
  {}

  virtual ~BlockVectorReader() {}
};

/*---------------------------------------------------------------------------*/

class LocalBlockVectorReader
: public Common::BlockVectorReaderT<Arccore::Real, Parameters<LocalIndexer>>
{
 public:
  LocalBlockVectorReader(const VectorData& vector)
  : Common::BlockVectorReaderT<Arccore::Real, Parameters<LocalIndexer>>(vector)
  {}

  virtual ~LocalBlockVectorReader() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
