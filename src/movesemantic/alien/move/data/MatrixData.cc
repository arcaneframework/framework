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

#include "MatrixData.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/data/ISpace.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixData::MatrixData()
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(0), std::make_shared<Space>(0),
                             std::make_shared<MatrixDistribution>(MatrixDistribution())))
{}

/*---------------------------------------------------------------------------*/

MatrixData::MatrixData(const Space& space, const MatrixDistribution& dist)
: m_impl(new MultiMatrixImpl(
  std::make_shared<Space>(space), std::make_shared<Space>(space), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

MatrixData::MatrixData(
const Space& row_space, const Space& col_space, const MatrixDistribution& dist)
: m_impl(new MultiMatrixImpl(
  std::make_shared<Space>(row_space), std::make_shared<Space>(col_space), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

MatrixData::MatrixData(Integer size, const MatrixDistribution& dist)
: m_impl(new MultiMatrixImpl(
  std::make_shared<Space>(size), std::make_shared<Space>(size), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

MatrixData::MatrixData(Integer row_size, Integer col_size, const MatrixDistribution& dist)
: m_impl(new MultiMatrixImpl(
  std::make_shared<Space>(row_size), std::make_shared<Space>(col_size), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

MatrixData::MatrixData(const MatrixDistribution& dist)
: m_impl(new MultiMatrixImpl(dist.rowDistribution().space().clone(),
                             dist.colDistribution().space().clone(), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

MatrixData::MatrixData(MatrixData&& matrix)
: m_impl(std::move(matrix.m_impl))
{}

/*---------------------------------------------------------------------------*/

MatrixData& MatrixData::operator=(MatrixData&& matrix)
{
  m_impl = std::move(matrix.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void MatrixData::init(const Space& space, const MatrixDistribution& dist)
{
  m_impl.reset(new MultiMatrixImpl(
  std::make_shared<Space>(space), std::make_shared<Space>(space), dist.clone()));
}

/*---------------------------------------------------------------------------*/

const Block*
MatrixData::block() const
{
  return m_impl->block();
}

/*---------------------------------------------------------------------------*/

const VBlock*
MatrixData::vblock() const
{
  return m_impl->vblock();
}

/*---------------------------------------------------------------------------*/

void MatrixData::setBlockInfos(const Integer block_size)
{
  m_impl->setBlockInfos(block_size);
}

/*---------------------------------------------------------------------------*/

void MatrixData::setBlockInfos(const Block* block)
{
  if (block) {
    m_impl->setBlockInfos(block);
  }
}

/*---------------------------------------------------------------------------*/

void MatrixData::setBlockInfos(const VBlock* block)
{
  if (block) {
    m_impl->setBlockInfos(block);
  }
}

/*---------------------------------------------------------------------------*/

void MatrixData::free()
{
  m_impl->free();
}

/*---------------------------------------------------------------------------*/

void MatrixData::clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/

void MatrixData::visit(ICopyOnWriteMatrix& m) const
{
  m.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const ISpace&
MatrixData::rowSpace() const
{
  return m_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/

const ISpace&
MatrixData::colSpace() const
{
  return m_impl->colSpace();
}

/*---------------------------------------------------------------------------*/

const MatrixDistribution&
MatrixData::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

void MatrixData::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool MatrixData::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool MatrixData::isComposite() const
{
  return m_impl->hasFeature("composite");
}

/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
MatrixData::impl()
{
  if (!m_impl) {
    m_impl.reset(new MultiMatrixImpl());
  }

  return m_impl.get();
}

/*---------------------------------------------------------------------------*/

const MultiMatrixImpl*
MatrixData::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/

MatrixData
MatrixData::clone() const
{
  MatrixData out;
  out.m_impl.reset(m_impl->clone());
  return out;
}

MatrixData createMatrixData(std::shared_ptr<MultiMatrixImpl> multi)
{
  MatrixData out;
  out.m_impl = multi;
  return out;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
