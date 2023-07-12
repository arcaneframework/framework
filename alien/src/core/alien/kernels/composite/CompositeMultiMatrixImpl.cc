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

#include "CompositeMultiMatrixImpl.h"

#include <alien/kernels/composite/CompositeMatrix.h>
#include <alien/kernels/composite/CompositeSpace.h>

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

  MultiMatrixImpl::MultiMatrixImpl()
  : std::tuple<std::shared_ptr<Space>, std::shared_ptr<Space>>{ std::make_shared<Space>(),
                                                                std::make_shared<Space>() }
  , Alien::MultiMatrixImpl(std::get<0>(*this), std::get<1>(*this),
                           std::make_shared<MatrixDistribution>(MatrixDistribution()))
  {
    insert("composite", new Matrix(this));
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
