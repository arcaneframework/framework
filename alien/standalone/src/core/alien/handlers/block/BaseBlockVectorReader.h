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

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/data/IVector.h>
#include <alien/data/utils/Parameters.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class SimpleCSRVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  template <typename ValueT, typename Parameters>
  class BlockVectorReaderT
  {
   public:
    using ValueType = ValueT;

   private:
    using Indexer = typename Parameters::Indexer;

   public:
    explicit BlockVectorReaderT(const IVector& vector);

    virtual ~BlockVectorReaderT() = default;

    ConstArrayView<ValueT> operator[](Integer iIndex) const
    {
      const Integer id = Indexer::index(iIndex, m_local_offset);
      if (this->m_block)
        return this->m_values.subConstView(
        id * this->m_block->size(), this->m_block->size());
      else if (this->m_vblock) {
        const VBlock* block_sizes = this->m_vblock;
        const Integer size = block_sizes->size(id);
        const Integer offset = this->m_vector_impl->vblockImpl().offset(id);
        return this->m_values.subConstView(offset, size);
      }
      else
        throw FatalErrorException(A_FUNCINFO, "No block info");
    }

   private:
    ConstArrayView<ValueT> m_values;
    const SimpleCSRVector<ValueT>* m_vector_impl;
    const Block* m_block;
    const VBlock* m_vblock;
    Integer m_local_offset;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "BlockVectorReaderT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
