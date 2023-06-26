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

#include <alien/handlers/block/BaseBlockVectorWriter.h>

#include <alien/ref/data/block/BlockVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BlockVectorWriter : public Common::BlockVectorWriterT<Real>
{
 public:
  BlockVectorWriter(BlockVector& vector)
  : Common::BlockVectorWriterT<Arccore::Real>(vector)
  {}

  virtual ~BlockVectorWriter() {}

  using Common::BlockVectorWriterT<Real>::operator=;
};

/*---------------------------------------------------------------------------*/

class LocalBlockVectorWriter : public Common::LocalBlockVectorWriterT<Real>
{
 public:
  LocalBlockVectorWriter(BlockVector& vector)
  : Common::LocalBlockVectorWriterT<Arccore::Real>(vector)
  {}

  virtual ~LocalBlockVectorWriter() {}

  using Common::LocalBlockVectorWriterT<Real>::operator=;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
