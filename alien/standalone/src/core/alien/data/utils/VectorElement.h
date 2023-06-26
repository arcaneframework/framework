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
 * \file VectorElement.h
 * \brief VectorElement.h
 */

#pragma once

#include <alien/utils/Precomp.h>

#include <arccore/collections/Array2.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 *
 * \brief Tool to manipulate a vector entry while building a vector
 * \tparam T The data type of the vector
 * \tparam Indexer The indexer
 */
template <typename T, typename Indexer>
class VectorElementT
{
 public:
  /*!
   * \brief Constructor
   * \param[in] values The array values
   * \param[in] indexes The indexes to work on
   * \param[in] local_offset The offset
   */
  VectorElementT(Arccore::ArrayView<T> values,
                 Arccore::ConstArrayView<Arccore::Integer> indexes,
                 Arccore::Integer local_offset);

  /*!
   * \brief Operator equal
   * \param[in] values The values to set
   */
  void operator=(Arccore::ConstArrayView<T> values);

  /*!
   * \brief Plus equal operator
   * \param[in] values The values to add
   */
  void operator+=(Arccore::ConstArrayView<T> values);

  /*!
   * \brief Minus equal operator
   * \param[in] values The values to substract
   */
  void operator-=(Arccore::ConstArrayView<T> values);

 private:
  //! The array of values
  Arccore::ArrayView<T>& m_values;
  //! The array of indexes
  Arccore::ConstArrayView<Arccore::Integer> m_indexes;
  //! The offset
  Arccore::Integer m_local_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 *
 * \brief Tool to manipulate and scale a vector entry while building a vector
 * \tparam T The data type of the vector
 * \tparam Indexer The indexer
 */
template <typename T, typename Indexer>
class MultVectorElementT
{
 public:
  /*!
   * \brief Constructor
   * \param[in] values The array values
   * \param[in] factor The factor to scale
   * \param[in] indexes The indexes to work on
   * \param[in] local_offset The offset
   */
  MultVectorElementT(Arccore::ArrayView<T> values, T factor,
                     Arccore::ConstArrayView<Arccore::Integer> indexes,
                     Arccore::Integer local_offset);

  /*!
   * \brief Operator equal
   * \param[in] values The values to set
   */
  void operator=(Arccore::ConstArrayView<T> values);

  /*!
   * \brief Operator plus equal
   * \param[in] values The values to add
   */
  void operator+=(Arccore::ConstArrayView<T> values);

  /*!
   * \brief Minus equal operator
   * \param[in] values The values to substract
   */
  void operator-=(Arccore::ConstArrayView<T> values);

 private:
  //! The array of values
  Arccore::ArrayView<T>& m_values;
  //! The scale factor
  T m_factor;
  //! The array of indexes
  Arccore::ConstArrayView<Arccore::Integer> m_indexes;
  //! The offset
  Arccore::Integer m_local_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 *
 * \brief Tool to manipulate and scale a vector entry while building a block vector
 * \tparam T The data type of the vector
 * \tparam Indexer The indexer
 */
template <typename T, typename Indexer>
class MultVectorElement2T
{
 public:
  /*!
   * \brief Constructor
   * \param[in] values The array values
   * \param[in] factor The factor to scale
   * \param[in] indexes The indexes to work on
   * \param[in] i The block entry
   * \param[in] local_offset The offset
   */
  MultVectorElement2T(Arccore::ArrayView<T> values,
                      T factor,
                      Arccore::ConstArray2View<Arccore::Integer> indexes,
                      Arccore::Integer i,
                      Arccore::Integer local_offset);

  /*!
   * \brief Operator equal
   * \param[in] values The values to set
   */
  void operator=(Arccore::ConstArray2View<T> values);

  /*!
   * \brief Operator plus equal
   * \param[in] values The values to add
   */
  void operator+=(Arccore::ConstArray2View<T> values);

  /*!
   * \brief Minus equal operator
   * \param[in] values The values to substract
   */
  void operator-=(Arccore::ConstArray2View<T> values);

 private:
  //! The array of values
  Arccore::ArrayView<T>& m_values;
  //! The scale factor
  T m_factor;
  //! The array of indexes
  Arccore::ConstArray2View<Arccore::Integer> m_indexes;
  //! The block index
  Arccore::Integer m_i;
  //! The offset
  Arccore::Integer m_local_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "VectorElementT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
