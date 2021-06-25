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

#include <alien/ref/data/scalar/Vector.h>

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Vector::Vector()
: m_impl(new MultiVectorImpl(std::make_shared<Space>(0),
                             std::make_shared<VectorDistribution>(VectorDistribution())))
{}

/*---------------------------------------------------------------------------*/

Vector::Vector(const VectorDistribution& dist)
: m_impl(new MultiVectorImpl(dist.space().clone(), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

Vector::Vector(Integer nrows, Integer nrows_local, IMessagePassingMng* parallel_mng)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(nrows),
                             std::make_shared<VectorDistribution>(
                             VectorDistribution(nrows, nrows_local, parallel_mng))))
{}

/*---------------------------------------------------------------------------*/

Vector::Vector(Integer nrows, IMessagePassingMng* parallel_mng)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(nrows),
                             std::make_shared<VectorDistribution>(VectorDistribution(nrows, parallel_mng))))
{}

/*---------------------------------------------------------------------------*/

Vector::Vector(Vector&& vector)
: m_impl(std::move(vector.m_impl))
{}

/*---------------------------------------------------------------------------*/

Vector&
Vector::operator=(Vector&& vector)
{
  m_impl = std::move(vector.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void Vector::visit(ICopyOnWriteVector& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const ISpace&
Vector::space() const
{
  return m_impl->space();
}

/*---------------------------------------------------------------------------*/

const VectorDistribution&
Vector::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

void Vector::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool Vector::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

MultiVectorImpl*
Vector::impl()
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
Vector::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
