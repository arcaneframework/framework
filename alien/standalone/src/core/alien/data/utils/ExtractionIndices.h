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
 * \file ExtractionIndices.h
 * \brief ExtractionIndices.h
 */

#pragma once

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 * \brief Tool to iterate over a matrix
 */
class ALIEN_EXPORT ExtractionIndices final
{
 public:
  /*!
   * \brief Constructor
   * \param[in] rowStart The index of the starting row
   * \param[in] rowRange The length of the row
   * \param[in] colStart The index of the starting col
   * \param[in] colRange The length of the col
   */
  ExtractionIndices(const Arccore::Integer rowStart, const Arccore::Integer rowRange,
                    const Arccore::Integer colStart = -1, const Arccore::Integer colRange = -1);

  //! Free resources
  ~ExtractionIndices();

  ExtractionIndices(const ExtractionIndices&) = delete;
  ExtractionIndices(ExtractionIndices&&) = delete;
  ExtractionIndices& operator=(const ExtractionIndices&) = delete;
  ExtractionIndices& operator=(ExtractionIndices&&) = delete;

  /*!
   * \brief Get the index of the starting row
   * \returns The index of the starting row
   */
  Arccore::Integer rowStart() const;

  /*!
   * \brief Get the length of the row
   * \returns The length of the row
   */
  Arccore::Integer rowRange() const;

  /*!
   * \brief Get the index of the starting col
   * \returns The index of the starting col
   */
  Arccore::Integer colStart() const;

  /*!
   * \brief Get the length of the col
   * \returns The length of the col
   */
  Arccore::Integer colRange() const;

  /*!
   * \brief Get the local index of a row given a global index
   * \param[in] uid The global index of a row
   * \returns The local index of a row
   */
  Arccore::Integer toLocalRow(const Arccore::Integer uid) const;

  /*!
   * \brief Get the local index of a col given a global index
   * \param[in] uid The global index of a col
   * \returns The local index of a col
   */
  Arccore::Integer toLocalCol(const Arccore::Integer uid) const;

 private:
  //! Index of the starting row
  Arccore::Integer m_starting_row;
  //! Index of the starting col
  Arccore::Integer m_starting_col;
  //! Length of the row
  Arccore::Integer m_row_range;
  //! Length of the col
  Arccore::Integer m_col_range;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
