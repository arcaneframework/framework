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
 * \file Block.h
 * \brief Block.h
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
 * \brief Block elements for block matrices
 *
 * Defines block parameters, for square or rectangulars block entries in a block matrix.
 * All blocks in the matrix are identical in term of size.
 */
class ALIEN_EXPORT Block final
{
 public:
  /*!
   * \brief Square block constructor
   * \param[in] block_size Size of the block
   */
  explicit Block(Arccore::Integer block_size);

  /*!
   * \brief Rectangular block constructor
   * \param[in] block_sizeX Size of the block in "X" direction
   * \param[in] block_sizeY Size of the block in "Y" direction
   */
  Block(Arccore::Integer block_sizeX, Arccore::Integer block_sizeY);

  /*!
   * \brief Copy constructor
   * \param[in] block The block to copy
   */
  Block(const Block& block);

  //! Free resources
  ~Block() = default;

  Block(Block&& block) = delete;
  Block& operator=(const Block& block) = delete;

 public:
  /*!
   * \brief Get square block size
   * \returns The size of the block
   */
  Arccore::Integer size() const;

  /*!
   * \brief Get rectangular block size in the "X" direction
   * \returns Block size in the "X" direction
   */
  Arccore::Integer sizeX() const;

  /*!
   * \brief Get rectangular block size in the "Y" direction
   * \returns Block size in the "Y" direction
   */
  Arccore::Integer sizeY() const;

  /*!
   * \brief Clone this object
   * \returns A copy of this object
   */
  std::shared_ptr<Block> clone() const;

 private:
  struct Internal;
  //! Actual implementation
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
