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

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/core/impl/MultiMatrixImpl.h>

#include <alien/kernels/dok/DoKDistributor.h>
#include <alien/kernels/dok/DoKLocalMatrixT.h>

#include <alien/kernels/redistributor/Redistributor.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

class DoKMatrix : public IMatrixImpl
{
 public:
  typedef Real ValueType;

 public:
  DoKMatrix(const MultiMatrixImpl* multi_impl = nullptr)
  : IMatrixImpl(multi_impl, "DoK")
  , m_data()
  {}

  DoKMatrix(const DoKMatrix&) = delete;

  virtual ~DoKMatrix() {}

  //! Demande la lib�ration des donn�es
  void clear() override {}

  bool setMatrixValue(Int32 row, Int32 col, const ValueType& value)
  {
    m_data.set(row, col, value);
    return true;
  }

  void assemble() { _distribute(); }

  void compact() { m_data.compact(); }

  DoKLocalMatrixT<ValueType>& data() { return m_data; }

  DoKLocalMatrixT<ValueType>& data() const { return m_data; }

  void dump() { m_data.dump(); }

 private:
  void _distribute()
  {
    Redistributor redist(
        distribution().globalRowSize(), distribution().parallelMng(), true);
    DoKDistributor dist(redist.commPlan());
    DoKLocalMatrixT<ValueType> new_data;
    // distribute does not work if src == tgt.
    dist.distribute(m_data, new_data);
    m_data = new_data;
    m_data.compact();
  }

 private:
  // TODO remove mutable !
  mutable DoKLocalMatrixT<ValueType> m_data;
};

} // namespace Alien
