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

#include "NullVector.h"

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NullVector::NullVector()
: m_space(0)
{}

/*---------------------------------------------------------------------------*/

void NullVector::visit(ICopyOnWriteVector& v ALIEN_UNUSED_PARAM) const
{
  throw FatalErrorException("NullVector can't be used in an expression");
}

/*---------------------------------------------------------------------------*/

const Space&
NullVector::space() const
{
  return m_space;
}

/*---------------------------------------------------------------------------*/

const VectorDistribution&
NullVector::distribution() const
{
  return m_distribution;
}

/*---------------------------------------------------------------------------*/

MultiVectorImpl*
NullVector::impl()
{
  // throw FatalErrorException("NullVector impl can't be used");
  return nullptr;
}

/*---------------------------------------------------------------------------*/

const MultiVectorImpl*
NullVector::impl() const
{
  // throw FatalErrorException("NullVector impl can't be used");
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
