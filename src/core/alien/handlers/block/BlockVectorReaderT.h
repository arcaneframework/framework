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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT, typename Parameters>
  BlockVectorReaderT<ValueT, Parameters>::BlockVectorReaderT(const IVector& vector)
  : m_vector_impl(NULL)
  , m_block(vector.impl()->block()) // Exception si mauvais kind
  , m_vblock(vector.impl()->vblock())
  , m_local_offset(0)
  {
    using namespace Alien;
    const SimpleCSRVector<ValueT>& v = vector.impl()->get<BackEnd::tag::simplecsr>();
    m_values = v.fullValues();
    m_vector_impl = &v;
    m_local_offset = v.distribution().offset();
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
