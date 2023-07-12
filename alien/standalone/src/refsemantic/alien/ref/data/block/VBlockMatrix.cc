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

#include <alien/ref/data/block/VBlockMatrix.h>

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockMatrix::VBlockMatrix()
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(0), std::make_shared<Space>(0),
                             std::make_shared<MatrixDistribution>(MatrixDistribution())))
{}

/*---------------------------------------------------------------------------*/

VBlockMatrix::VBlockMatrix(const VBlock& block, const MatrixDistribution& dist)
: m_impl(
  new MultiMatrixImpl(dist.rowSpace().clone(), dist.colSpace().clone(), dist.clone()))
{
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

VBlockMatrix::VBlockMatrix(
const VBlock& row_block, const VBlock& col_block, const MatrixDistribution& dist)
: m_impl(
  new MultiMatrixImpl(dist.rowSpace().clone(), dist.colSpace().clone(), dist.clone()))
{
  m_impl->setRowBlockInfos(&row_block);
  m_impl->setColBlockInfos(&col_block);
}

/*---------------------------------------------------------------------------*/

VBlockMatrix::VBlockMatrix(Integer nrows, Integer ncols, Integer nrows_local,
                           const VBlock& row_block, const VBlock& col_block, IMessagePassingMng* parallel_mng)
: m_impl(
  new MultiMatrixImpl(std::make_shared<Space>(nrows), std::make_shared<Space>(ncols),
                      std::make_shared<MatrixDistribution>(
                      MatrixDistribution(nrows, ncols, nrows_local, parallel_mng))))
{
  m_impl->setRowBlockInfos(&row_block);
  m_impl->setColBlockInfos(&col_block);
}

/*---------------------------------------------------------------------------*/

VBlockMatrix::VBlockMatrix(Integer nrows, Integer ncols, const VBlock& row_block,
                           const VBlock& col_block, IMessagePassingMng* parallel_mng)
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(nrows),
                             std::make_shared<Space>(ncols),
                             std::make_shared<MatrixDistribution>(MatrixDistribution(nrows, ncols, parallel_mng))))
{
  m_impl->setRowBlockInfos(&row_block);
  m_impl->setColBlockInfos(&col_block);
}

/*---------------------------------------------------------------------------*/

VBlockMatrix::VBlockMatrix(VBlockMatrix&& matrix)
: m_impl(std::move(matrix.m_impl))
{}

/*---------------------------------------------------------------------------*/

VBlockMatrix&
VBlockMatrix::operator=(VBlockMatrix&& matrix)
{
  m_impl = std::move(matrix.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void VBlockMatrix::init(const VBlock& block, const MatrixDistribution& dist)
{
  m_impl.reset(new MultiMatrixImpl(
  dist.rowSpace().clone(), dist.colSpace().clone(), dist.clone()));
  m_impl->setBlockInfos(&block);
}

/*---------------------------------------------------------------------------*/

void VBlockMatrix::free()
{
  m_impl->free();
}

/*---------------------------------------------------------------------------*/

void VBlockMatrix::clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/

void VBlockMatrix::visit(ICopyOnWriteMatrix& m) const
{
  m.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const MatrixDistribution&
VBlockMatrix::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

const ISpace&
VBlockMatrix::rowSpace() const
{
  return m_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/

const ISpace&
VBlockMatrix::colSpace() const
{
  return m_impl->colSpace();
}

/*---------------------------------------------------------------------------*/

void VBlockMatrix::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool VBlockMatrix::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

const VBlock&
VBlockMatrix::vblock() const
{
  const VBlock* block = m_impl->vblock();
  if (block)
    return *block;
  else
    throw FatalErrorException(
    A_FUNCINFO, "Requesting for block information but none was provided");
}

/*---------------------------------------------------------------------------*/

const VBlock&
VBlockMatrix::rowBlock() const
{
  const VBlock* block = m_impl->rowBlock();
  if (block)
    return *block;
  else
    throw FatalErrorException(
    A_FUNCINFO, "Requesting for block information but none was provided");
}

/*---------------------------------------------------------------------------*/

const VBlock&
VBlockMatrix::colBlock() const
{
  const VBlock* block = m_impl->colBlock();
  if (block)
    return *block;
  else
    throw FatalErrorException(
    A_FUNCINFO, "Requesting for block information but none was provided");
}

/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
VBlockMatrix::impl()
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
VBlockMatrix::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
