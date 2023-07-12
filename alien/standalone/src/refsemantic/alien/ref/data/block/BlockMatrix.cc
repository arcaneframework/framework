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

#include <alien/ref/data/block/BlockMatrix.h>

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockMatrix::BlockMatrix()
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(0), std::make_shared<Space>(0),
                             std::make_shared<MatrixDistribution>(MatrixDistribution())))
{}

/*---------------------------------------------------------------------------*/

BlockMatrix::BlockMatrix(const Block& block, const MatrixDistribution& dist)
: m_impl(
  new MultiMatrixImpl(dist.rowSpace().clone(), dist.colSpace().clone(), dist.clone()))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

BlockMatrix::BlockMatrix(Integer nrows, Integer ncols, Integer nrows_local,
                         const Block& block, IMessagePassingMng* parallel_mng)
: m_impl(
  new MultiMatrixImpl(std::make_shared<Space>(nrows), std::make_shared<Space>(ncols),
                      std::make_shared<MatrixDistribution>(
                      MatrixDistribution(nrows, ncols, nrows_local, parallel_mng))))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

BlockMatrix::BlockMatrix(
Integer nrows, Integer ncols, const Block& block, IMessagePassingMng* parallel_mng)
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(nrows),
                             std::make_shared<Space>(ncols),
                             std::make_shared<MatrixDistribution>(MatrixDistribution(nrows, ncols, parallel_mng))))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

BlockMatrix::BlockMatrix(BlockMatrix&& matrix)
: m_impl(std::move(matrix.m_impl))
{}

/*---------------------------------------------------------------------------*/

BlockMatrix&
BlockMatrix::operator=(BlockMatrix&& matrix)
{
  m_impl = std::move(matrix.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void BlockMatrix::init(const Block& block, const MatrixDistribution& dist)
{
  m_impl.reset(new MultiMatrixImpl(
  dist.rowSpace().clone(), dist.colSpace().clone(), dist.clone()));
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

void BlockMatrix::free()
{
  m_impl->free();
}

/*---------------------------------------------------------------------------*/

void BlockMatrix::clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/

void BlockMatrix::visit(ICopyOnWriteMatrix& m) const
{
  m.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const MatrixDistribution&
BlockMatrix::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

const ISpace&
BlockMatrix::rowSpace() const
{
  return m_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/

const ISpace&
BlockMatrix::colSpace() const
{
  return m_impl->colSpace();
}

/*---------------------------------------------------------------------------*/

void BlockMatrix::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool BlockMatrix::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

const Block&
BlockMatrix::block() const
{
  const Block* block = m_impl->block();
  if (block)
    return *block;
  else
    throw FatalErrorException(
    A_FUNCINFO, "Requesting for block information but none was provided");
}

/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
BlockMatrix::impl()
{
  if (!m_impl) {
    m_impl.reset(new MultiMatrixImpl());
  }
  /* JMG ????
     else if (!m_impl.unique()) { // Need to clone due to other references.
     m_impl.reset(m_impl->clone());
     } */
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/

const MultiMatrixImpl*
BlockMatrix::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
