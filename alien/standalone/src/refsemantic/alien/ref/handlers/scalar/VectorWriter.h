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

#include <alien/ref/data/scalar/Vector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorWriter
: public Common::VectorWriterT<Arccore::Real, Parameters<GlobalIndexer>>
{
 private:
  using Base = Common::VectorWriterT<Real, Parameters<GlobalIndexer>>;

 public:
  VectorWriter(Vector& vector)
  : Base(vector)
  {}

  virtual ~VectorWriter() {}

  using Base::operator=;
};

/*---------------------------------------------------------------------------*/

class LocalVectorWriter
: public Common::VectorWriterT<Arccore::Real, Parameters<LocalIndexer>>
{
 private:
  using Base = Common::VectorWriterT<Real, Parameters<LocalIndexer>>;

 public:
  LocalVectorWriter(Vector& vector)
  : Base(vector)
  {}

  virtual ~LocalVectorWriter() {}

  using Base::operator=;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
