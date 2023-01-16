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

#include "VectorData.h"

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/utils/ICopyOnWriteObject.h>
#include <alien/data/ISpace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{
using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorData::VectorData()
: m_impl(new MultiVectorImpl(std::make_shared<Space>(0),
                             std::make_shared<VectorDistribution>(VectorDistribution())))
{}

/*---------------------------------------------------------------------------*/

VectorData::VectorData(const ISpace& space, const VectorDistribution& dist)
: m_impl(new MultiVectorImpl(space.clone(), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

VectorData::VectorData(Integer size, const VectorDistribution& dist)
: m_impl(new MultiVectorImpl(std::make_shared<Space>(size), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

VectorData::VectorData(const VectorDistribution& dist)
: m_impl(new MultiVectorImpl(dist.space().clone(), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

VectorData::VectorData(VectorData&& vector)
: m_impl(std::move(vector.m_impl))
{}

/*---------------------------------------------------------------------------*/

VectorData& VectorData::operator=(VectorData&& vector)
{
  m_impl = std::move(vector.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void VectorData::init(const ISpace& space, const VectorDistribution& dist)
{
  m_impl.reset(new MultiVectorImpl(space.clone(), dist.clone()));
}

/*---------------------------------------------------------------------------*/

void VectorData::setBlockInfos(const Integer block_size)
{
  impl()->setBlockInfos(block_size);
}

/*---------------------------------------------------------------------------*/
/*
void
VectorData::
setBlockInfos(const IBlockBuilder& builder)
{
  std::unique_ptr<Block> block(new Block(m_impl->distribution(), builder.blockSizes()));
  impl()->setBlockInfos(std::move(block));
}*/

/*---------------------------------------------------------------------------*/

void VectorData::setBlockInfos(const Block* block)
{
  if (block) {
    impl()->setBlockInfos(block);
  }
}

/*---------------------------------------------------------------------------*/

void VectorData::setBlockInfos(const VBlock* block)
{
  if (block) {
    impl()->setBlockInfos(block);
  }
}

/*---------------------------------------------------------------------------*/

void VectorData::free()
{
  impl()->free();
}

/*---------------------------------------------------------------------------*/

void VectorData::clear()
{
  impl()->clear();
}

/*---------------------------------------------------------------------------*/

void VectorData::visit(ICopyOnWriteVector& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const ISpace&
VectorData::space() const
{
  return impl()->space();
}

/*---------------------------------------------------------------------------*/

const VectorDistribution&
VectorData::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

const Block*
VectorData::block() const
{
  return m_impl->block();
}

/*---------------------------------------------------------------------------*/

const VBlock*
VectorData::vblock() const
{
  return m_impl->vblock();
}

/*---------------------------------------------------------------------------*/

void VectorData::setUserFeature(String feature)
{
  impl()->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool VectorData::hasUserFeature(String feature) const
{
  return impl()->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

MultiVectorImpl*
VectorData::impl()
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
VectorData::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/

VectorData
VectorData::clone() const
{
  VectorData out;
  out.m_impl.reset(m_impl->clone());
  return out;
}

VectorData createVectorData(std::shared_ptr<MultiVectorImpl> multi)
{
  VectorData out;
  out.m_impl = multi;
  return out;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
