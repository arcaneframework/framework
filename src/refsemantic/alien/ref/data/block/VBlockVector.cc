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

#include <alien/ref/data/block/VBlockVector.h>

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockVector::VBlockVector()
: m_impl(new MultiVectorImpl(std::make_shared<Space>(0),
                             std::make_shared<VectorDistribution>(VectorDistribution())))
{}

/*---------------------------------------------------------------------------*/

VBlockVector::VBlockVector(const VBlock& block, const VectorDistribution& dist)
: m_impl(new MultiVectorImpl(dist.space().clone(), dist.clone()))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

VBlockVector::VBlockVector(Integer nrows, Integer nrows_local, const VBlock& block,
                           IMessagePassingMng* parallel_mng)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(nrows),
                             std::make_shared<VectorDistribution>(
                             VectorDistribution(nrows, nrows_local, parallel_mng))))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

VBlockVector::VBlockVector(
Integer nrows, const VBlock& block, IMessagePassingMng* parallel_mng)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(nrows),
                             std::make_shared<VectorDistribution>(VectorDistribution(nrows, parallel_mng))))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

VBlockVector::VBlockVector(VBlockVector&& vector)
: m_impl(std::move(vector.m_impl))
{}

/*---------------------------------------------------------------------------*/

VBlockVector&
VBlockVector::operator=(VBlockVector&& vector)
{
  m_impl = std::move(vector.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void VBlockVector::init(const VBlock& block, const VectorDistribution& dist)
{
  m_impl.reset(new MultiVectorImpl(dist.space().clone(), dist.clone()));
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

void VBlockVector::free()
{
  m_impl->free();
}

/*---------------------------------------------------------------------------*/

void VBlockVector::clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/

const ISpace&
VBlockVector::space() const
{
  return m_impl->space();
}

/*---------------------------------------------------------------------------*/

void VBlockVector::visit(ICopyOnWriteVector& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const VectorDistribution&
VBlockVector::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

void VBlockVector::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool VBlockVector::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

const VBlock&
VBlockVector::vblock() const
{
  const VBlock* block = m_impl->vblock();
  if (block)
    return *block;
  else
    throw FatalErrorException(
    A_FUNCINFO, "Requesting for block information but none was provided");
}

/*---------------------------------------------------------------------------*/

MultiVectorImpl*
VBlockVector::impl()
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
VBlockVector::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
