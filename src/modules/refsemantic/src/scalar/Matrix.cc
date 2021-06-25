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

#include <alien/ref/data/scalar/Matrix.h>

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix::Matrix()
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(0), std::make_shared<Space>(0),
                             std::make_shared<MatrixDistribution>(MatrixDistribution())))
{}

/*---------------------------------------------------------------------------*/

Matrix::Matrix(const MatrixDistribution& dist)
: m_impl(
  new MultiMatrixImpl(dist.rowSpace().clone(), dist.colSpace().clone(), dist.clone()))
{}

/*---------------------------------------------------------------------------*/

Matrix::Matrix(
Integer nrows, Integer ncols, Integer nrows_local, IMessagePassingMng* parallel_mng)
: m_impl(
  new MultiMatrixImpl(std::make_shared<Space>(nrows), std::make_shared<Space>(ncols),
                      std::make_shared<MatrixDistribution>(
                      MatrixDistribution(nrows, ncols, nrows_local, parallel_mng))))
{}

/*---------------------------------------------------------------------------*/

Matrix::Matrix(Integer nrows, Integer ncols, IMessagePassingMng* parallel_mng)
: m_impl(new MultiMatrixImpl(std::make_shared<Space>(nrows),
                             std::make_shared<Space>(ncols),
                             std::make_shared<MatrixDistribution>(MatrixDistribution(nrows, ncols, parallel_mng))))
{}

/*---------------------------------------------------------------------------*/

Matrix::Matrix(Matrix&& matrix)
: m_impl(std::move(matrix.m_impl))
{}

/*---------------------------------------------------------------------------*/

Matrix::~Matrix() {}

/*---------------------------------------------------------------------------*/

Matrix&
Matrix::operator=(Matrix&& matrix)
{
  m_impl = std::move(matrix.m_impl);
  return *this;
}

/*---------------------------------------------------------------------------*/

void Matrix::visit(ICopyOnWriteMatrix& m) const
{
  m.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const MatrixDistribution&
Matrix::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

const ISpace&
Matrix::rowSpace() const
{
  return m_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/

const ISpace&
Matrix::colSpace() const
{
  return m_impl->colSpace();
}

/*---------------------------------------------------------------------------*/

void Matrix::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool Matrix::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
Matrix::impl()
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
Matrix::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
