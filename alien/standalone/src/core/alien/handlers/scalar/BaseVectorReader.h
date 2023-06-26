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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT, typename Parameters>
  class VectorReaderT
  {
   public:
    using ValueType = ValueT;

   private:
    using Indexer = typename Parameters::Indexer;

   public:
    explicit VectorReaderT(const IVector& vector);

    virtual ~VectorReaderT() = default;

    inline const ValueType& operator[](Integer iIndex) const
    {
      const Integer id = Indexer::index(iIndex, m_local_offset);

      return m_values[id];
    }

    [[nodiscard]] inline Arccore::Integer size() const { return m_values.size(); }

   private:
    ConstArrayView<ValueType> m_values;

    Integer m_local_offset;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  using GlobalVectorReader = Common::VectorReaderT<double, Parameters<GlobalIndexer>>;
  using LocalVectorReader = Common::VectorReaderT<double, Parameters<LocalIndexer>>;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "VectorReaderT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
