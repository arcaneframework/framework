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

#include "NullMatrix.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NullMatrix::NullMatrix()
: m_space(0)
{}

/*---------------------------------------------------------------------------*/

void NullMatrix::visit(ICopyOnWriteMatrix& m ALIEN_UNUSED_PARAM) const
{
  throw FatalErrorException("NullMatrix can't be used in an expression");
}

/*---------------------------------------------------------------------------*/

const MatrixDistribution&
NullMatrix::distribution() const
{
  return m_distribution;
}

/*---------------------------------------------------------------------------*/

const Space&
NullMatrix::rowSpace() const
{
  return m_space;
}

/*---------------------------------------------------------------------------*/

const Space&
NullMatrix::colSpace() const
{
  return m_space;
}

/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
NullMatrix::impl()
{
  // throw FatalErrorException("NullMatrix impl can't be used");
  return nullptr;
}

/*---------------------------------------------------------------------------*/

const MultiMatrixImpl*
NullMatrix::impl() const
{
  // throw FatalErrorException("NullMatrix impl can't be used");
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
