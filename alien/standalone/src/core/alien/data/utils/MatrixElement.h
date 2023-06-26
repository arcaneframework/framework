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

/*!
 * \file MatrixElement.h
 * \brief MatrixElement.h
 */

#ifndef ALIEN_COMMON_UTILS_MATRIXELEMENT_H
#define ALIEN_COMMON_UTILS_MATRIXELEMENT_H

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 * \brief Tool to manipulate a matrix entry while building the matrix
 * \tparam Builder The type of the builder used to build the matrix
 */
template <typename Builder>
class MatrixElementT
{
 public:
  /*!
   * \brief Constructor
   * \param[in] iIndex The row index of the entry
   * \param[in] jIndex The col index of the entry
   * \param[in] parent The builder used to build the matrix
   */
  MatrixElementT(
  const Arccore::Integer iIndex, const Arccore::Integer jIndex, Builder& parent)
  : m_iIndex(iIndex)
  , m_jIndex(jIndex)
  , m_parent(parent)
  {}

  /*!
   * \brief accessor operator
   * \return value
   */
  Arccore::Real operator()() const { return m_parent.getData(m_iIndex, m_jIndex); }

  /*!
   * \brief Add and set operator
   * \param[in] value The value to add
   */
  void operator+=(Real value)
  {
    m_parent.addData(m_iIndex, m_jIndex, value);
  }

  /*!
   * \brief Minus and set operator
   * \param[in] value The value to substract
   */
  void operator-=(Real value)
  {
    m_parent.addData(m_iIndex, m_jIndex, -value);
  }

  /*!
   * \brief Assignment operator
   * \param[in] value The value to set
   */
  void operator=(Real value)
  {
    m_parent.setData(m_iIndex, m_jIndex, value);
  }

  /*!
   * \brief Comparison operator
   * \param[in] other To be compare against.
   */
  template <typename Builder2>
  bool operator=(const MatrixElementT<Builder2>& other)
  {
    bool test_pattern = (m_iIndex == other.m_iIndex) && (m_jIndex == other.m_jIndex);
    // TODO: Check values.
    return test_pattern;
  }

 private:
  //! The row index
  const Arccore::Integer m_iIndex;
  //! The col index
  const Arccore::Integer m_jIndex;
  //! The builder
  Builder& m_parent;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_COMMON_UTILS_MATRIXELEMENT_H */
