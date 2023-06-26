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

#include <alien/ref/data/scalar/RedistributedVector.h>

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RedistributedVector::RedistributedVector(IVector& vector, Redistributor& redist)
: m_impl(redist.redistribute(vector.impl()))
{}

RedistributedVector::~RedistributedVector() {}

/*---------------------------------------------------------------------------*/

void RedistributedVector::visit(ICopyOnWriteVector& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const VectorDistribution&
RedistributedVector::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

const ISpace&
RedistributedVector::space() const
{
  return m_impl->space();
}

/*---------------------------------------------------------------------------*/

void RedistributedVector::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool RedistributedVector::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

MultiVectorImpl*
RedistributedVector::impl()
{
  if (!m_impl) {
    m_impl.reset(new MultiVectorImpl());
  }
  /* JMG ????
  else if (!m_impl.unique()) { // Need to clone due to other references.
    m_impl.reset(m_impl->clone());
  } */
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/

const MultiVectorImpl*
RedistributedVector::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
