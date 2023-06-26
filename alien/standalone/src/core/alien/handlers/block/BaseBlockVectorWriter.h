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

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#include <alien/data/IVector.h>

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

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  class BlockVectorWriterBaseT
  {
   public:
    explicit BlockVectorWriterBaseT(IVector& vector);

    virtual ~BlockVectorWriterBaseT() { end(); }

    void end();

    BlockVectorWriterBaseT& operator=(const ValueT v);

   protected:
    IVector& m_vector;
    ArrayView<ValueT> m_values;
    SimpleCSRVector<ValueT>* m_vector_impl;
    const Block* m_block;
    const VBlock* m_vblock;
    bool m_finalized;
    bool m_changed;
  };

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  class LocalBlockVectorWriterT : public BlockVectorWriterBaseT<ValueT>
  {
   public:
    explicit LocalBlockVectorWriterT(IVector& vector);

    ~LocalBlockVectorWriterT() = default;

    using BlockVectorWriterBaseT<ValueT>::operator=;

    ArrayView<ValueT> operator[](Integer iIndex)
    {
      this->m_changed = true;
      if (this->m_block)
        return this->m_values.subView(
        iIndex * this->m_block->size(), this->m_block->size());
      else if (this->m_vblock) {
        const VBlock* block_sizes = this->m_vblock;
        const Integer size = block_sizes->size(iIndex);
        const Integer offset = this->m_vector_impl->vblockImpl().offset(iIndex);
        return this->m_values.subView(offset, size);
      }
      else
        throw FatalErrorException(A_FUNCINFO, "No block infos");
    }
  };

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  class BlockVectorWriterT : public BlockVectorWriterBaseT<ValueT>
  {
   public:
    explicit BlockVectorWriterT(IVector& vector);

    ~BlockVectorWriterT() = default;

    using BlockVectorWriterBaseT<ValueT>::operator=;

    ArrayView<ValueT> operator[](Integer iIndex)
    {
      this->m_changed = true;
      if (this->m_block)
        return this->m_values.subView(
        (iIndex - m_local_offset) * this->m_block->size(), this->m_block->size());
      else if (this->m_vblock) {
        const VBlock* block_sizes = this->m_vblock;
        // const Integer size = block_sizes.size(iIndex-m_local_offset);
        const Integer size = block_sizes->size(iIndex);
        const Integer offset = this->m_vector_impl->vblockImpl().offset(iIndex);
        return this->m_values.subView(offset, size);
      }
      else
        throw FatalErrorException(A_FUNCINFO, "No block infos");
    }

   private:
    Integer m_local_offset;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  using BlockVectorWriter = BlockVectorWriterT<Real>;
  using LocalBlockVectorWriter = LocalBlockVectorWriterT<Real>;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "BlockVectorWriterT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
