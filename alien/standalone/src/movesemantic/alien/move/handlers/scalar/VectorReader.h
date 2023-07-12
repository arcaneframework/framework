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

#include <alien/handlers/scalar/BaseVectorReader.h>

// FIXME: Check if correct object is used.
#include <alien/move/data/VectorData.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorReader
: public Common::VectorReaderT<Arccore::Real, Parameters<GlobalIndexer>>
{
 public:
  VectorReader(const VectorData& vector)
  : Common::VectorReaderT<Arccore::Real, Parameters<GlobalIndexer>>(vector)
  {}

  virtual ~VectorReader() = default;
};

/*---------------------------------------------------------------------------*/

class LocalVectorReader
: public Common::VectorReaderT<Arccore::Real, Parameters<LocalIndexer>>
{
 public:
  LocalVectorReader(const VectorData& vector)
  : Common::VectorReaderT<Arccore::Real, Parameters<LocalIndexer>>(vector)
  {}

  virtual ~LocalVectorReader() = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
