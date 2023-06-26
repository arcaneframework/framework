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

/*!
 * \file alien/data/CompositeMatrix.cc
 * \brief CompositeMatrix.cc
 */

#include <alien/data/CompositeMatrix.h>
#include <alien/kernels/composite/CompositeMatrix.h>
#include <alien/kernels/composite/CompositeMultiMatrixImpl.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/functional/NullMatrix.h>
#include <alien/kernels/composite/CompositeSpace.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeMatrix::Element
CompositeElement(CompositeMatrix& m, Integer i, Integer j)
{
  return m.composite(i, j);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeMatrix::CompositeMatrix()
: CompositeMatrix(0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeMatrix::CompositeMatrix(Integer nc)
: m_impl(new CompositeKernel::MultiMatrixImpl())
, m_composite_matrix(m_impl->get<Alien::BackEnd::tag::composite>(false))
{
  m_impl->setFeature("composite");

  resize(nc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeMatrix::visit(ICopyOnWriteMatrix& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeMatrix::resize(Integer nc)
{
  m_composite_matrix.resize(nc);

  for (Integer i = 0; i < nc; ++i)
    for (Integer j = 0; j < nc; ++j)
      m_composite_matrix.setComposite(i, j, new NullMatrix());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
CompositeMatrix::size() const
{
  return m_composite_matrix.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
CompositeMatrix::rowSpace() const
{
  return m_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
CompositeMatrix::colSpace() const
{
  return m_impl->colSpace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeMatrix::Element
CompositeMatrix::composite(Integer i, Integer j)
{
  return m_composite_matrix.element(i, j);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMatrix&
CompositeMatrix::operator()(Integer i, Integer j)
{
  return m_composite_matrix(i, j);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const IMatrix&
CompositeMatrix::operator()(Integer i, Integer j) const
{
  return m_composite_matrix(i, j);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeMatrix::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CompositeMatrix::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
CompositeMatrix::impl()
{
  if (!m_impl) {
    m_impl.reset(new CompositeKernel::MultiMatrixImpl());
  }
  // TOCHECK : needs to be removed or not ?
  /* JMG ????
     else if (!m_impl.unique()) { // Need to clone due to other references.
     m_impl.reset(m_impl->clone());
     } */
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const MultiMatrixImpl*
CompositeMatrix::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeMatrix::free()
{
  for (Integer i = 0; i < this->size(); ++i)
    for (Integer j = 0; j < this->size(); ++j) {
      if (m_composite_matrix(i, j).impl())
        m_composite_matrix(i, j).impl()->free();
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeMatrix::clear()
{
  for (Integer i = 0; i < this->size(); ++i)
    for (Integer j = 0; j < this->size(); ++j) {
      if (m_composite_matrix(i, j).impl())
        m_composite_matrix(i, j).impl()->clear();
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
