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

class IMatrix;
class ISpace;
class Timestamp;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT MatrixElement
  {
   public:
    MatrixElement(std::shared_ptr<IMatrix>& element, std::shared_ptr<ISpace>& row_space,
                  std::shared_ptr<ISpace>& col_space, Timestamp& timestamp)
    : m_element(element)
    , m_row_space(row_space)
    , m_col_space(col_space)
    , m_timestamp(timestamp)
    {}
    template <typename T>
    void operator=(T&& v) { _assign(new T(std::move(v))); }

   private:
    void _assign(IMatrix* matrix);

   private:
    std::shared_ptr<IMatrix>& m_element;
    std::shared_ptr<ISpace>& m_row_space;
    std::shared_ptr<ISpace>& m_col_space;
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
