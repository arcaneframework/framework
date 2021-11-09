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
 * \file BlockSizeVector.h
 * \brief BlockSizeVector.h
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
 * \ingroup block
 * \brief Block elements for block vectors
 *
 * Defines block parameters, for block entries in a block vector.
 */
class ALIEN_EXPORT BlockSizeVector
{
 public:
  /*!
   * \brief Constructor
   * \param[in] sizes The size of the different blocks
   * \param[in] offset The distribution offset
   * \param[in] indexes Entries indexes
   */
  BlockSizeVector(Arccore::UniqueArray<Arccore::Integer>& sizes, Arccore::Integer offset,
                  Arccore::ConstArrayView<Arccore::Integer> indexes);

  /*!
   * \brief Operator equal
   * \param[in] size Set the size of the blocks
   */
  BlockSizeVector& operator=(Arccore::Integer size);

  /*!
   * \brief Operator plus equal
   * \param[in] size Add the size to the blocks
   */
  BlockSizeVector& operator+=(Arccore::Integer size);

 private:
  //! Sizes of the blocks
  Arccore::UniqueArray<Arccore::Integer>& m_sizes;
  //! Distribution offset
  Arccore::Integer m_offset;
  //! Entries indexes
  Arccore::ConstArrayView<Arccore::Integer> m_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup block
 * \brief Block vector filler
 *
 * Defines block parameters, for non uniform block entries in a block vector.
 */
class ALIEN_EXPORT BlockSizeVectorFiller
{
  //! Constructor
 public:
  BlockSizeVectorFiller()
  : m_offset(0)
  {}

  /*!
   * \brief Operator bracket
   * \param[in] indexes List of indexes to access
   * \returns Block to access
   */
  BlockSizeVector operator[](Arccore::ConstArrayView<Arccore::Integer> indexes);

  /*!
   * \brief Operator bracket
   * \param[in] indexes List of indexes to access
   * \returns Block to access
   */
  BlockSizeVector operator[](Arccore::ConstArray2View<Arccore::Integer> indexes);

  /*!
   * \brief Operator bracket
   * \param[in] index Index to access
   * \returns The block size at the specified index
   */
  Arccore::Integer& operator[](Arccore::Integer index);

 protected:
  //! Sizes in the block vector
  Arccore::UniqueArray<Arccore::Integer> m_sizes;
  //! Distribution offset
  Arccore::Integer m_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
