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

#pragma once

#include <vector>

#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/IVector.h>
#include <alien/kernels/composite/CompositeBackEnd.h>
#include <alien/kernels/composite/CompositeSpace.h>
#include <alien/kernels/composite/CompositeVectorElement.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class MultiVectorImpl;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT Vector : public IVectorImpl
  {
   public:
    //! Constructor, from a composite.
    Vector(MultiVectorImpl* multi_impl);

    //! Constructor, converting data.
    //! @throw not implemented yet.
    Vector(const Alien::MultiVectorImpl* multi_impl);

    void init(const VectorDistribution& dist, bool need_allocate);

    void clear();

    void free();

    void resize(Integer size);

    Integer size() const;

    VectorElement element(Integer i);

    IVector& operator[](Integer i);

    const IVector& operator[](Integer i) const;

    void setComposite(Integer i, IVector* v);

   private:
    std::vector<std::shared_ptr<IVector>> m_vectors;

    Space m_space;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
