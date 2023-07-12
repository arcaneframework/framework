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
 * \file alien/data/CompositeVector.cc
 * \brief CompositeVector.cc
 */

#include <alien/data/CompositeVector.h>
#include <alien/kernels/composite/CompositeMultiVectorImpl.h>
#include <alien/kernels/composite/CompositeVector.h>

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/functional/NullVector.h>
#include <alien/kernels/composite/CompositeSpace.h>
#include <alien/utils/ICopyOnWriteObject.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeVector::Element
CompositeElement(CompositeVector& v, Integer i)
{
  return v.composite(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeVector::CompositeVector()
: CompositeVector(0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeVector::CompositeVector(Integer nc)
: m_impl(new CompositeKernel::MultiVectorImpl())
, m_composite_vector(m_impl->get<Alien::BackEnd::tag::composite>(false))
{
  m_impl->setFeature("composite");

  if (nc > 0)
    resize(nc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeVector::visit(ICopyOnWriteVector& v) const
{
  v.accept(m_impl);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeVector::resize(Integer size)
{
  m_composite_vector.resize(size);

  for (Integer i = 0; i < size; ++i)
    m_composite_vector.setComposite(i, new NullVector());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
CompositeVector::size() const
{
  return m_composite_vector.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
CompositeVector::space() const
{
  return m_impl->space();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CompositeVector::Element
CompositeVector::composite(Integer i)
{
  return m_composite_vector.element(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVector&
CompositeVector::operator[](Integer i)
{
  return m_composite_vector[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const IVector&
CompositeVector::operator[](Integer i) const
{
  return m_composite_vector[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeVector::setUserFeature(String feature)
{
  m_impl->setFeature(feature);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CompositeVector::hasUserFeature(String feature) const
{
  return m_impl->hasFeature(feature);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiVectorImpl*
CompositeVector::impl()
{
  if (!m_impl) {
    m_impl.reset(new MultiVectorImpl());
  }
  // TOCHECK : to remove or not ?
  /* JMG ????
     else if (!m_impl.unique()) { // Need to clone due to other references.
     m_impl.reset(m_impl->clone());
     } */
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const MultiVectorImpl*
CompositeVector::impl() const
{
  return m_impl.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeVector::free()
{
  for (Integer i = 0; i < this->size(); ++i)
    if (m_composite_vector[i].impl())
      m_composite_vector[i].impl()->free();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CompositeVector::clear()
{
  for (Integer i = 0; i < this->size(); ++i)
    if (m_composite_vector[i].impl())
      m_composite_vector[i].impl()->clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
