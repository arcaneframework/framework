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

#include <alien/data/IVector.h>
#include <alien/data/utils/Parameters.h>
#include <alien/data/utils/VectorElement.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Timestamp;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  class VectorWriterBaseT
  {
   public:
    typedef ValueT ValueType;

   public:
    VectorWriterBaseT(IVector& vector, bool update = true);

    virtual ~VectorWriterBaseT() { end(); }

    void end();

    void operator=(const ValueType v);

   protected:
    ArrayView<ValueType> m_values;
    Timestamp* m_time_stamp;
    Integer m_local_offset;
    bool m_finalized;
  };

  /*---------------------------------------------------------------------------*/

  template <typename ValueT, typename Parameters>
  class VectorWriterT : public VectorWriterBaseT<ValueT>
  {
   public:
    using ValueType = ValueT;

   private:
    using Indexer = typename Parameters::Indexer;

   public:
    using VectorElement = VectorElementT<ValueT, Indexer>;
    using MultVectorElement = MultVectorElementT<ValueT, Indexer>;
    using MultVectorElement2 = MultVectorElement2T<ValueT, Indexer>;

    using VectorWriterBaseT<ValueT>::operator=;

    VectorWriterT(IVector& vector);

    virtual ~VectorWriterT() {}

    inline ValueType& operator[](Integer iIndex)
    {
      const Integer id = Indexer::index(iIndex, this->m_local_offset);
      return this->m_values[id];
    }

    inline Integer size() const { return this->m_values.size(); }

    inline VectorElement operator()(ConstArrayView<Integer> indexes)
    {
      return VectorElement(this->m_values, indexes, this->m_local_offset);
    }

    inline MultVectorElement operator()(
    ValueType factor, ConstArrayView<Integer> indexes)
    {
      return MultVectorElement(this->m_values, factor, indexes, this->m_local_offset);
    }

    inline MultVectorElement2 operator()(
    ValueType factor, ConstArray2View<Integer> indexes, Integer i)
    {
      return MultVectorElement2(this->m_values, factor, indexes, i, this->m_local_offset);
    }
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "VectorWriterT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
