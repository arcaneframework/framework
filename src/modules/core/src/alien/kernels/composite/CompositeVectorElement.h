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

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVector;
class ISpace;
class Timestamp;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT VectorElement
  {
   public:
    VectorElement(std::shared_ptr<IVector>& element, std::shared_ptr<ISpace>& space,
                  Timestamp& timestamp)
    : m_element(element)
    , m_space(space)
    , m_timestamp(timestamp)
    {}
    template <typename T>
    void operator=(T&& v) { _assign(new T(std::move(v))); }

   private:
    void _assign(IVector* vector);

   private:
    std::shared_ptr<IVector>& m_element;
    std::shared_ptr<ISpace>& m_space;
    Timestamp& m_timestamp;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
