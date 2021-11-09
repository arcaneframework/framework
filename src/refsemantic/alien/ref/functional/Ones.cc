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

#include <alien/ref/functional/Ones.h>

#include <alien/ref/data/block/BlockVector.h>
#include <alien/ref/data/block/VBlockVector.h>
#include <alien/ref/data/scalar/Vector.h>
#include <alien/ref/handlers/block/BlockVectorWriter.h>
#include <alien/ref/handlers/block/VBlockVectorWriter.h>
#include <alien/ref/handlers/scalar/VectorWriter.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Vector
ones(Integer size, IMessagePassingMng* pm)
{
  Vector v(size, pm);
  {
    Alien::VectorWriter w(v);
    w = 1.;
  }
  return v;
}

/*---------------------------------------------------------------------------*/

BlockVector
ones(Integer size, const Block& bloc, IMessagePassingMng* pm)
{
  BlockVector v(size, bloc, pm);
  const Integer offset = v.distribution().offset();
  {
    Alien::BlockVectorWriter w(v);
    for (Integer i = offset; i < v.distribution().localSize() + offset; ++i) {
      ArrayView<Real> values = w[i];
      for (Integer j = 0; j < bloc.size(); ++j)
        values[j] = 1.;
    }
  }
  return v;
}

/*---------------------------------------------------------------------------*/

VBlockVector
ones(Integer size, const VBlock& bloc, IMessagePassingMng* pm)
{
  VBlockVector v(size, bloc, pm);
  const Integer offset = v.distribution().offset();
  {
    Alien::VBlockVectorWriter w(v);
    for (Integer i = offset; i < v.distribution().localSize() + offset; ++i) {
      ArrayView<Real> values = w[i];
      for (Integer j = 0; j < bloc.size(i); ++j)
        values[j] = 1.;
    }
  }
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien
