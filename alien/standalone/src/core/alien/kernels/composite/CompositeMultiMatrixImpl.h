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

#include <memory>
#include <tuple>

#include <alien/core/impl/MultiMatrixImpl.h>

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

  class Space;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  struct ObjectWithTwoCompositeSpaces
  {
    ObjectWithTwoCompositeSpaces() = default;

    std::shared_ptr<Space> m_composite_row_space;
    std::shared_ptr<Space> m_composite_col_space;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT MultiMatrixImpl
  : private std::tuple<std::shared_ptr<Space>, std::shared_ptr<Space>>
  , public Alien::MultiMatrixImpl
  {
   public:
    MultiMatrixImpl();

    virtual ~MultiMatrixImpl() {}

    Space& rowSpace() { return *std::get<0>(*this).get(); }
    Space& colSpace() { return *std::get<1>(*this).get(); }
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
