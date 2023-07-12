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

#include <alien/ref/data/block/BlockVector.h>

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockVector::BlockVector()
: m_impl(new MultiVectorImpl(std::make_shared<Space>(0),
                             std::make_shared<VectorDistribution>(VectorDistribution())))
{}

/*---------------------------------------------------------------------------*/

BlockVector::BlockVector(const Block& block, const VectorDistribution& dist)
: m_impl(new MultiVectorImpl(dist.space().clone(), dist.clone()))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

BlockVector::BlockVector(Integer nrows, Integer nrows_local, const Block& block,
                         IMessagePassingMng* parallel_mng)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(nrows),
                             std::make_shared<VectorDistribution>(
                             VectorDistribution(nrows, nrows_local, parallel_mng))))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

BlockVector::BlockVector(
Integer nrows, const Block& block, IMessagePassingMng* parallel_mng)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(nrows),
                             std::make_shared<VectorDistribution>(VectorDistribution(nrows, parallel_mng))))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

BlockVector::BlockVector(BlockVector&& vector)
: m_impl(std::move(vector.m_impl))
{}

/*---------------------------------------------------------------------------*/

BlockVector&
BlockVector::operator=(BlockVector&& vector)
{
  m_impl = std::move(vector.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void BlockVector::init(const Block& block, const VectorDistribution& dist)
{
  m_impl.reset(new MultiVectorImpl(dist.space().clone(), dist.clone()));
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

void BlockVector::free()
{
  m_impl->free();
}

/*---------------------------------------------------------------------------*/

void BlockVector::clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/

const ISpace&
BlockVector::space() const
{
  return m_impl->space();
}

/*---------------------------------------------------------------------------*/

void BlockVector::visit(ICopyOnWriteVector& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const VectorDistribution&
BlockVector::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

void BlockVector::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool BlockVector::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

const Block&
BlockVector::block() const
{
  const Block* block = m_impl->block();
  if (block)
    return *block;
  else
    throw FatalErrorException(
    A_FUNCINFO, "Requesting for block information but none was provided");
}

/*---------------------------------------------------------------------------*/

MultiVectorImpl*
BlockVector::impl()
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
BlockVector::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
