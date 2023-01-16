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

#include "CompositeVector.h"

#include <alien/kernels/composite/CompositeMultiVectorImpl.h>
#include <alien/kernels/composite/CompositeSpace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  Vector::Vector(const Alien::MultiVectorImpl* multi_impl)
  : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::composite>::name())
  {
    alien_fatal([&] {
      cout() << "CompositeVector(const Alien::MultiVectorImpl*) : Not implemented";
    });
  }

  /*---------------------------------------------------------------------------*/

  Vector::Vector(MultiVectorImpl* multi_impl)
  : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::composite>::name())
  , m_space(multi_impl->space())
  {
    alien_debug([&] { cout() << "Construct CompositeVector " << this; });
  }

  /*---------------------------------------------------------------------------*/

  void Vector::init(const VectorDistribution& dist ALIEN_UNUSED_PARAM,
                    bool need_allocate ALIEN_UNUSED_PARAM)
  {
    alien_debug([&] { cout() << "Initializing CompositeVector " << this; });
  }

  /*---------------------------------------------------------------------------*/

  void Vector::clear()
  {
    alien_debug([&] { cout() << "Clear CompositeVector" << this; });

    for (auto& v : m_vectors) {
      if (v->impl())
        v->impl()->clear();
    }
  }

  /*---------------------------------------------------------------------------*/

  void Vector::free()
  {
    alien_debug([&] { cout() << "Free CompositeVector " << this; });

    for (auto& v : m_vectors) {
      if (v->impl())
        v->impl()->free();
    }
  }

  /*---------------------------------------------------------------------------*/

  void Vector::resize(Integer nc)
  {
    alien_debug([&] {
      cout() << "Resize CompositeVector " << this;
      cout() << " - old size = " << m_vectors.size();
      cout() << " - new size = " << nc;
    });

    m_space.resizeSubSpace(nc);

    m_vectors.resize(nc);
  }

  /*---------------------------------------------------------------------------*/

  Integer Vector::size() const { return m_vectors.size(); }

  /*---------------------------------------------------------------------------*/

  VectorElement Vector::element(Integer i)
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    return VectorElement(m_vectors[i], m_space[i], *this);
  }

  /*---------------------------------------------------------------------------*/

  IVector& Vector::operator[](Integer i)
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    return *m_vectors[i];
  }

  /*---------------------------------------------------------------------------*/

  const IVector& Vector::operator[](Integer i) const
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    return *m_vectors[i];
  }

  /*---------------------------------------------------------------------------*/

  void Vector::setComposite(Integer i, IVector* v)
  {
    ALIEN_ASSERT(i < size(), "Bound error");
    m_vectors[i].reset(v);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
