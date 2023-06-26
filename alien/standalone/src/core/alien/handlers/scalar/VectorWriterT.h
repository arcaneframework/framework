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

  template <typename ValueT>
  VectorWriterBaseT<ValueT>::VectorWriterBaseT(IVector& vector, bool update)
  : m_time_stamp(nullptr)
  , m_local_offset(0)
  , m_finalized(false)
  {
    auto& v = vector.impl()->get<BackEnd::tag::simplecsr>(update);
    m_local_offset = v.distribution().offset();
    m_values = v.fullValues();
    m_time_stamp = &v;
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void VectorWriterBaseT<ValueT>::end()
  {
    if (m_finalized)
      return;
    m_finalized = true;
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void VectorWriterBaseT<ValueT>::operator=(const ValueType v)
  {
    for (Integer i = 0, is = m_values.size(); i < is; ++i)
      m_values[i] = v;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT, typename Parameters>
  VectorWriterT<ValueT, Parameters>::VectorWriterT(IVector& vector)
  : VectorWriterBaseT<ValueT>(vector)
  {}

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
