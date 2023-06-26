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

namespace Alien
{

/*!
 * Matrix stored as Dictionary Of Keys
 */
class DoKMatrix : public IMatrixImpl
{
 public:
  typedef Real ValueType;

 public:
  explicit DoKMatrix(const MultiMatrixImpl* multi_impl = nullptr)
  : IMatrixImpl(multi_impl, "DoK")
  , m_data()
  {}

  DoKMatrix(const DoKMatrix&) = delete;

  ~DoKMatrix() override = default;

  void clear() override {}

  //! Set value of a matrix element, creating it if it does not exist yet.
  //! \param row id of the row in the matrix
  //! \param col id of the column in the matrix
  //! \param value value of this non-zero
  //! \return
  bool setNNZ(Int32 row, Int32 col, const ValueType& value)
  {
    m_data.set(row, col, value);
    m_need_update = true;
    return true;
  }

  //! Set value of a matrix element, creating it if it does not exist yet.
  //! \param row id of the row in the matrix
  //! \param col id of the column in the matrix
  //! \param value value of this non-zero
  //! \return
  ValueType addNNZ(Int32 row, Int32 col, const ValueType& value)
  {
    m_need_update = true;
    return m_data.add(row, col, value);
  }

  //! Dispatch matrix elements
  void assemble() { _distribute(); }

  void compact() { m_data.compact(); }

  DoKLocalMatrixT<ValueType>& data() { return m_data; }

  DoKLocalMatrixT<ValueType>& data() const { return m_data; }

  void dump() { m_data.dump(); }

 private:
  void _distribute()
  {
    m_need_update = Arccore::MessagePassing::mpAllReduce(distribution().parallelMng(), Arccore::MessagePassing::ReduceSum, m_need_update);
    if (!m_need_update) {
      return;
    }
    Redistributor redist(
    distribution().globalRowSize(), distribution().parallelMng(), distribution().parallelMng());
    DoKDistributor dist(redist.commPlan());
    DoKLocalMatrixT<ValueType> new_data;
    // distribute does not work if src == tgt.
    dist.distribute(m_data, new_data);
    m_data = new_data;
    m_data.compact();
    m_need_update = false;
  }

 private:
  // TODO remove mutable !
  mutable DoKLocalMatrixT<ValueType> m_data;
  bool m_need_update = true;
};

} // namespace Alien
