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

#include <arccore/message_passing/Messages.h>

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
  BlockVectorWriterBaseT<ValueT>::BlockVectorWriterBaseT(IVector& vector)
  : m_vector(vector)
  , m_vector_impl(nullptr)
  , m_block(vector.impl()->block()) // Exception si mauvais kind
  , m_vblock(vector.impl()->vblock())
  , m_finalized(false)
  , m_changed(false)
  {
    using namespace Alien;
    SimpleCSRVector<ValueT>& v =
    m_vector.impl()->template get<BackEnd::tag::simplecsr>(false);
    m_values = v.fullValues();
    m_vector_impl = &v;
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  LocalBlockVectorWriterT<ValueT>::LocalBlockVectorWriterT(IVector& vector)
  : BlockVectorWriterBaseT<ValueT>(vector)
  {}

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void BlockVectorWriterBaseT<ValueT>::end()
  {
    IMessagePassingMng* parallelMng = m_vector_impl->distribution().parallelMng();
    m_changed = Arccore::MessagePassing::mpAllReduce(
    parallelMng, Arccore::MessagePassing::ReduceMax, m_changed);
    if (m_finalized or not m_changed)
      return;
    m_finalized = true;
    m_vector_impl->updateTimestamp();
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  BlockVectorWriterBaseT<ValueT>& BlockVectorWriterBaseT<ValueT>::operator=(
  const ValueT v)
  {
    m_values.fill(v);
    return *this;
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  BlockVectorWriterT<ValueT>::BlockVectorWriterT(IVector& vector)
  : BlockVectorWriterBaseT<ValueT>(vector)
  , m_local_offset(0) // On ne peut plus utiliser vector!!
  {
    using Base = BlockVectorWriterBaseT<ValueT>;
    m_local_offset = Base::m_vector.impl()->distribution().offset();
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
