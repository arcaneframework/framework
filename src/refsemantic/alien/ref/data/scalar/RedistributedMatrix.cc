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

#include <alien/ref/data/scalar/RedistributedMatrix.h>

#include <alien/core/impl/MultiMatrixImpl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RedistributedMatrix::RedistributedMatrix(IMatrix& matrix, Redistributor& redist)
: m_impl(redist.redistribute(matrix.impl()))
{}

RedistributedMatrix::~RedistributedMatrix() {}

/*---------------------------------------------------------------------------*/

void RedistributedMatrix::visit(ICopyOnWriteMatrix& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/

const MatrixDistribution&
RedistributedMatrix::distribution() const
{
  return m_impl->distribution();
}

/*---------------------------------------------------------------------------*/

const ISpace&
RedistributedMatrix::rowSpace() const
{
  return m_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/

const ISpace&
RedistributedMatrix::colSpace() const
{
  return m_impl->colSpace();
}

/*---------------------------------------------------------------------------*/

void RedistributedMatrix::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/

bool RedistributedMatrix::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
RedistributedMatrix::impl()
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
RedistributedMatrix::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
