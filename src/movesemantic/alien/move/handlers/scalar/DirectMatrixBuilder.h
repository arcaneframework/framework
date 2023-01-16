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

#include <alien/handlers/scalar/BaseDirectMatrixBuilder.h>
#include <alien/utils/MoveObject.h>

#include <alien/move/data/MatrixData.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectMatrixBuilder
{
 public:
  using MatrixElement = MatrixElementT<DirectMatrixBuilder>;

  DirectMatrixBuilder(MatrixData&& matrix, const DirectMatrixOptions::ResetFlag reset_flag,
                      const DirectMatrixOptions::SymmetricFlag symmetric_flag = DirectMatrixOptions::SymmetricFlag::eSymmetric)
  : m_data(std::move(matrix))
  , m_builder(m_data, reset_flag, symmetric_flag)
  {}

  virtual ~DirectMatrixBuilder() = default;

  MatrixData&& release()
  {
    m_builder.finalize();

    return std::move(m_data);
  }

  void reserve(Arccore::Integer n, DirectMatrixOptions::ReserveFlag flag = DirectMatrixOptions::ReserveFlag::eResetReservation)
  {
    m_builder.reserve(n, flag);
  }

  void reserve(Arccore::ConstArrayView<Arccore::Integer> indices, Arccore::Integer n,
               DirectMatrixOptions::ReserveFlag flag = DirectMatrixOptions::ReserveFlag::eResetReservation)
  {
    m_builder.reserve(indices, n, flag);
  }

  void allocate()
  {
    m_builder.allocate();
  }

  [[deprecated("Use contribute() instead.")]] MatrixElement operator()(const Integer iIndex, const Integer jIndex)
  {
    return MatrixElement(iIndex, jIndex, *this);
  }

  std::optional<Arccore::Real> contribute(Arccore::Integer iIndex, Arccore::Integer jIndex, Arccore::Real value)
  {
    m_builder.addData(iIndex, jIndex, value);
    return { value };
  }

  [[deprecated("Use contribute() instead.")]] void addData(Arccore::Integer iIndex, Arccore::Integer jIndex, Arccore::Real value)
  {
    m_builder.addData(iIndex, jIndex, value);
  }

  void addData(Arccore::Integer iIndex, Arccore::Real factor,
               Arccore::ConstArrayView<Arccore::Integer> jIndexes,
               Arccore::ConstArrayView<Arccore::Real> jValues)
  {
    m_builder.addData(iIndex, factor, jIndexes, jValues);
  }

  void setData(Arccore::Integer iIndex, Arccore::Integer jIndex, Arccore::Real value)
  {
    m_builder.setData(iIndex, jIndex, value);
  }

  void setData(Arccore::Integer iIndex, Arccore::Real factor,
               Arccore::ConstArrayView<Arccore::Integer> jIndexes,
               Arccore::ConstArrayView<Arccore::Real> jValues)
  {
    m_builder.setData(iIndex, factor, jIndexes, jValues);
  }

  void finalize()
  {
    m_builder.finalize();
  }

  void squeeze()
  {
    m_builder.squeeze();
  }

  [[nodiscard]] Arccore::String stats() const
  {
    return m_builder.stats();
  }
  [[nodiscard]] Arccore::String stats(Arccore::IntegerConstArrayView ids) const
  {
    return m_builder.stats(ids);
  }

 private:
  MatrixData m_data;
  Common::DirectMatrixBuilder m_builder;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
